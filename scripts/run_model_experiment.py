from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiment/src"))
from robotics_lab.refinement_v3 import main
if __name__=="__main__":main()
