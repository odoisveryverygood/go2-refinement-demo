"""Explicit OpenAI cloud-planner substitution with a small local spending cap.

Reads an already authorized process-local key. Never writes credentials or logs
request headers. Prices are the retrieved GPT-4.1 standard input/output rates;
cost is an estimate, not a billing statement. No automatic paid GPU provisioning.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[3]

class CloudChat:
    def __init__(self,out:Path,model='gpt-4.1-2025-04-14',budget_usd=1.0):
        if not os.environ.get('OPENAI_API_KEY'):raise RuntimeError('An authorized process-local OpenAI key is required')
        self.out=out;out.mkdir(parents=True,exist_ok=True)
        self.model=model;self.budget=budget_usd;self.calls=[];self.lock=threading.Lock()
        self.max_requests=int(os.environ.get('ROBOTICS_MAX_REQUESTS','128'))
        assert 1<=self.max_requests<=512
        self.ledger=ROOT/'evidence/cloud-pilot-cost-ledger.json'
        self.ledger.parent.mkdir(exist_ok=True)
        self.total=json.loads(self.ledger.read_text()) if self.ledger.exists() else {'requests':0,'estimated_usd':0.0,'input_tokens':0,'output_tokens':0}
        self.input_rate=2/1_000_000;self.output_rate=8/1_000_000
    def close(self):
        pass
    def __call__(self,body,timeout=90):
        with self.lock:return self.call(body,timeout)
    def call(self,body,timeout):
        messages=[];image_hashes=[];input_bytes=0
        for msg in body['messages']:
            text=str(msg['content']);input_bytes+=len(text.encode())
            if msg.get('images'):
                content=[{'type':'text','text':text}]
                for encoded in msg['images']:
                    raw=base64.b64decode(encoded,validate=True)
                    assert len(raw)<4*1024*1024
                    image_hashes.append(hashlib.sha256(raw).hexdigest())
                    mime='image/png' if raw.startswith(b'\x89PNG') else 'image/jpeg'
                    content.append({'type':'image_url','image_url':{'url':f'data:{mime};base64,{encoded}','detail':'low'}})
                messages.append({'role':msg['role'],'content':content})
            else:messages.append({'role':msg['role'],'content':text})
        cap=min(256,int(body.get('options',{}).get('num_predict',150)))
        # UTF-8 byte count bounds text-token count conservatively. Images here
        # are small simulated camera frames; reserve 16k tokens per image.
        reserve=(input_bytes+len(image_hashes)*16000)*self.input_rate+cap*self.output_rate
        assert self.total['requests']<self.max_requests,'Cloud pilot request cap reached'
        assert self.total['estimated_usd']+reserve<=self.budget,'Cloud pilot spending cap reached'
        payload={'model':self.model,'messages':messages,'temperature':float(body.get('options',{}).get('temperature',0.0)),
            'max_completion_tokens':cap,'store':False}
        if body.get('format')=='json':payload['response_format']={'type':'json_object'}
        submitted=time.monotonic_ns();request_id=len(self.calls)
        row={'request_id':request_id,'submitted_ns':submitted,'state':'running','requested_alias':body.get('model'),'global_request_cap':self.max_requests,'estimated_budget_usd':self.budget,
             'actual_provider_model':self.model,'image_sha256':image_hashes,
             'messages_without_images':[{k:v for k,v in msg.items() if k!='images'} for msg in body['messages']]}
        self.calls.append(row)
        request=urllib.request.Request('https://api.openai.com/v1/chat/completions',data=json.dumps(payload).encode(),
            headers={'Authorization':'Bearer '+os.environ['OPENAI_API_KEY'],'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(request,timeout=timeout) as response:
                data=json.load(response);provider_request_id=response.headers.get('x-request-id')
            usage=data.get('usage',{});inp=int(usage.get('prompt_tokens',0));out=int(usage.get('completion_tokens',0))
            cost=inp*self.input_rate+out*self.output_rate
            self.total['requests']+=1;self.total['input_tokens']+=inp;self.total['output_tokens']+=out;self.total['estimated_usd']+=cost
            self.ledger.write_text(json.dumps(self.total,indent=2)+'\n')
            row.update(state='completed',received_ns=time.monotonic_ns(),model=data.get('model'),
                message=data['choices'][0]['message'],usage=usage,estimated_usd=cost,
                provider_request_id=provider_request_id,finish_reason=data['choices'][0].get('finish_reason'))
        except urllib.error.HTTPError as error:
            try:details=json.loads(error.read()).get('error',{})
            except Exception:details={}
            row.update(state='failed',received_ns=time.monotonic_ns(),http_status=error.code,
                error_code=details.get('code'),error_type=details.get('type'),error_parameter=details.get('param'))
            with (self.out/'cloud-calls.jsonl').open('a') as file:file.write(json.dumps(row)+'\n')
            raise RuntimeError(f'OpenAI request failed: HTTP {error.code}, code {details.get("code")}, parameter {details.get("param")}') from None
        except Exception as error:
            row.update(state='failed',received_ns=time.monotonic_ns(),error_type=type(error).__name__,billing_usage_unknown=True)
            with (self.out/'cloud-calls.jsonl').open('a') as file:file.write(json.dumps(row)+'\n')
            raise RuntimeError('Cloud request did not complete; usage may need billing reconciliation') from None
        with (self.out/'cloud-calls.jsonl').open('a') as file:file.write(json.dumps(row)+'\n')
        return {'message':row['message'],'metrics':{k:v for k,v in row.items() if k not in ['message','messages_without_images']}}
