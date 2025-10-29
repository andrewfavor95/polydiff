import logging
LOGGER = logging.getLogger(__name__)
import assertpy
import subprocess
import unittest

from deepdiff import DeepDiff

import error
import benchmark.sweep_hyperparam
ic.configureOutput(includeContext=True)

class TestBenchmark(unittest.TestCase):

    def test_arg_combos(self):
        for arg_str, want in [
            ('''
        a=1|2|3
        ''',
        [
            {'a':'1'},
            {'a':'2'},
            {'a':'3'}
        ]
        ),
        ('''
        (a=1 b=2)|(a=3 b=4)
        ''',
        [
            {'a':'1', 'b':'2'},
            {'a':'3', 'b':'4'},
        ]),
        ('''
        c=5
        (a=1 b=2)|(a=3 b=4)
        ''',
        [
            {'c':'5','a':'1', 'b':'2'},
            {'c':'5','a':'3', 'b':'4'},
        ]),

        ('''
        a=1
        (b=2)|(a=3 b=4)
        ''',
        [
            {'a':'1', 'b':'2'},
            {'a':'3', 'b':'4'},
        ]),
        ('''
        a=1
        (a=2 b=3)|(b=4|5)
        ''',
        [
            {'a':'2', 'b':'3'},
            {'a':'1', 'b':'4'},
            {'a':'1', 'b':'5'},
        ]),
        ('''
        (a=1)|(b=4|5)
        ''',
        [
            {'a':'1'},
            {'b':'4'},
            {'b':'5'},
        ]),
        ('''
        (a=1)|((b=4)|(b=5))
        ''',
        [
            {'a':'1'},
            {'b':'4'},
            {'b':'5'},
        ]),
        # TODO: Implement check such that this testcase returns an error, as it is somewhat nonsensical.
        # ('''
        # arg1=A|B
        # (arg1=C arg3=D)|(arg1=c arg3=d)
        # ''',
        # ['error']
        # ),
        ]:
            with error.context(f'{arg_str=} {want=}'):
                got = benchmark.sweep_hyperparam.get_arg_combos(arg_str)
                LOGGER.debug(got, want)
                self.assertEqual(got, want)

    def test_subprocess_retcode_tee(self):
        '''This test just serves as documentation for the fact that when using tee, process returncodes are lost.
        
        This is why in the pipeline we do not use tee for outputting logs when running in_proc.
        '''
        job = '''python raise_exception.py | tee /home/ahern/benchmarks/aa_diffusion/tip_atoms/220420_tip_pilot_2/out/run_cond1_0.log'''
        print(f'running job: {job}')
        proc = subprocess.run(job, shell=True)
        print(f'{proc=}') 
        print(f'{proc.returncode=}')
        assertpy.assert_that(proc.returncode).is_equal_to(0)
    
    def test_subprocess_retcode(self):
        job = '''python raise_exception.py'''
        print(f'running job: {job}')
        proc = subprocess.run(job, shell=True)
        print(f'{proc=}') 
        print(f'{proc.returncode=}')
        assertpy.assert_that(proc.returncode).is_not_equal_to(0)

if __name__ == '__main__':
        unittest.main()