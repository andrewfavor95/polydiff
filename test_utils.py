import re
import json
from icecream import ic
import os
import subprocess
from pathlib import Path
import yaml
import unittest
import pickle

from rf2aa import tensor_util
golden_dir = 'goldens'


def golden_path(name, processor_specific=False):
    processor_name = 'any'
    if processor_specific:
        processor_name = processor()
    return Path(os.path.join(golden_dir, f'{name}_{processor_name}'))

def get_golden_path(name):
    p = golden_path()
    if not p.exists():
        raise Exception(f'golden at {p} does not exist, please generate it')
    return p


def write(path, got):
    with open(path, 'wb') as fh:
        pickle.dump(got, fh)

def read(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)

def assert_matches_golden(t, name, got, rewrite=False, processor_specific=False, custom_comparator=None):
    p = golden_path(name, processor_specific=processor_specific)
    if rewrite:
        print(f'rewriting {p}')
        write(p, got)
        # p.write_text(got)
        return
    if not p.exists():
        raise Exception(f'golden at {p} does not exist, please generate it')
    # want = p.read_text()
    want = read(p)
    if custom_comparator:
        diff = custom_comparator(got, want)
        if not diff:
            return
        try:
            jsoned = diff.to_json()
            loaded = json.loads(jsoned)
            fail_msg = json.dumps(loaded, indent=4)
        except Exception as e:
            ic('failed to pretty print output', e)
            fail_msg = json.dumps(diff.pop('tensors unequal', ''), indent=4) + '\n' + str(diff) 
        t.fail(fail_msg)
    else:
        t.assertEqual(got, want)

def assertEqual(t, custom_comparator, got, want):
    diff = custom_comparator(got, want)
    if not diff:
        return
    try:
        jsoned = diff.to_json()
        loaded = json.loads(jsoned)
        fail_msg = json.dumps(loaded, indent=4)
    except Exception as e:
        ic('failed to pretty print output', e)
        fail_msg = json.dumps(diff.pop('tensors unequal', ''), indent=4) + '\n' + str(diff) 
    t.fail(fail_msg)


def processor():
    o = subprocess.run(['lscpu | grep "Model name"'], capture_output=True, shell=True)
    s = o.stdout.decode("utf-8")
    s = s[len("Model name:"):].strip()
    ncpus = available_cpu_count()
    s = f'{s}-{ncpus}CPUs'
    return s

def available_cpu_count():
    """ Number of available virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling
    userspace-only program"""

    # cpuset
    # cpuset may restrict the number of *available* processors
    try:
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                      open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0:
                return res
    except IOError:
        pass

    # Python 2.6+
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        pass

    # https://github.com/giampaolo/psutil
    try:
        import psutil
        return psutil.cpu_count()   # psutil.NUM_CPUS on old versions
    except (ImportError, AttributeError):
        pass

    # POSIX
    try:
        res = int(os.sysconf('SC_NPROCESSORS_ONLN'))

        if res > 0:
            return res
    except (AttributeError, ValueError):
        pass

    # Windows
    try:
        res = int(os.environ['NUMBER_OF_PROCESSORS'])

        if res > 0:
            return res
    except (KeyError, ValueError):
        pass

    # jython
    try:
        from java.lang import Runtime
        runtime = Runtime.getRuntime()
        res = runtime.availableProcessors()
        if res > 0:
            return res
    except ImportError:
        pass

    # BSD
    try:
        sysctl = subprocess.Popen(['sysctl', '-n', 'hw.ncpu'],
                                  stdout=subprocess.PIPE)
        scStdout = sysctl.communicate()[0]
        res = int(scStdout)

        if res > 0:
            return res
    except (OSError, ValueError):
        pass

    # Linux
    try:
        res = open('/proc/cpuinfo').read().count('processor\t:')

        if res > 0:
            return res
    except IOError:
        pass

    # Solaris
    try:
        pseudoDevices = os.listdir('/devices/pseudo/')
        res = 0
        for pd in pseudoDevices:
            if re.match(r'^cpuid@[0-9]+$', pd):
                res += 1

        if res > 0:
            return res
    except OSError:
        pass

    # Other UNIXes (heuristic)
    try:
        try:
            dmesg = open('/var/run/dmesg.boot').read()
        except IOError:
            dmesgProcess = subprocess.Popen(['dmesg'], stdout=subprocess.PIPE)
            dmesg = dmesgProcess.communicate()[0]

        res = 0
        while '\ncpu' + str(res) + ':' in dmesg:
            res += 1

        if res > 0:
            return res
    except OSError:
        pass

    raise Exception('Can not determine number of CPUs on this system')

def cmp_pretty(got, want, **kwargs):
    diff = tensor_util.cmp(got, want, **kwargs)
    if not diff:
        return
    try:
        jsoned = diff.to_json()
        loaded = json.loads(jsoned)
        return json.dumps(loaded, indent=4)
    except Exception as e:
        ic('failed to pretty print output', e)
        return json.dumps(diff.pop('tensors unequal', ''), indent=4) + '\n' + str(diff)