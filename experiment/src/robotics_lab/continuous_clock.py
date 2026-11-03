"""Monotonic deadline clock including system suspend on macOS."""
import ctypes,sys,time
if sys.platform=='darwin':
    class Timebase(ctypes.Structure):
        _fields_=[('numer',ctypes.c_uint32),('denom',ctypes.c_uint32)]
    _library=ctypes.CDLL('/usr/lib/libSystem.B.dylib');_info=Timebase()
    _library.mach_timebase_info.argtypes=[ctypes.POINTER(Timebase)]
    assert _library.mach_timebase_info(ctypes.byref(_info))==0
    _library.mach_continuous_time.restype=ctypes.c_uint64
    def continuous_ns():return _library.mach_continuous_time()*_info.numer//_info.denom
else:
    def continuous_ns():
        return time.clock_gettime_ns(time.CLOCK_BOOTTIME) if hasattr(time,'CLOCK_BOOTTIME') else time.monotonic_ns()
def continuous_s():return continuous_ns()/1e9
