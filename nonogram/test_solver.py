import unittest
import numpy as np

from solver import solve_line, line_state_str


SOLVE_LINE_TEST_CASES = [
    {
        'line': np.full((5,), -1),
        'groups': [],
        'result': np.full((5,), 0),
        'skip': False,
    },
    {
        'line': np.full((7,), -1),
        'groups': [3],
        'result': np.full((7,), -1),
        'skip': False,
    },
    {
        'line': np.array([-1, 0, -1, -1, -1, -1]),
        'groups': [3],
        'result': [0, 0, -1, 1, 1, -1],
        'skip': False,
    },
    {
        'line': np.full((9,), -1),
        'groups': [1, 2, 3],
        'result': [-1, -1, -1, 1, -1, -1, 1, 1, -1],
        'skip': False,
    },
    {
        'line': np.array([-1, -1, -1, 1, 1, -1, -1, -1]),
        'groups': [3],
        'result': [0, 0, -1, 1, 1, -1, 0, 0],
        'skip': False,
    },
    {
        'line': np.array([-1, 1, -1, 1, 1, -1]),
        'groups': [1, 2],
        'result': [0, 1, 0, 1, 1, 0],
        'skip': False,
    },
    {
        'line': np.array([-1, -1, 1, -1, -1, -1]),
        'groups': [2, 1],
        'result': [0, -1, 1, -1, -1, -1],
        'skip': False,
    },
    {
        'line': np.array([-1, -1, -1, 0, -1, -1, -1]),
        'groups': [2, 2],
        'result': [-1, 1, -1, 0, -1, 1, -1],
        'skip': False,
    },
    {
        'line': np.array([1, 1, -1, -1, -1, -1]),
        'groups': [2, 1],
        'result': [1, 1, 0, -1, -1, -1],
        'skip': False,
    },
    {
        'line': np.array([-1, -1, -1, 1, 1, -1, -1, -1]),
        'groups': [2, 2],
        'result': [-1, -1, 0, 1, 1, 0, -1, -1],
        'skip': False,
    },
    {
        'line': np.array([-1, -1, -1, 1, 1, -1, -1, -1, -1]),
        'groups': [2, 3],
        'result': [-1, -1, -1, 1, 1, -1, -1, -1, -1],
        'skip': False,
    },
    {
        'line': np.array([0, 0, 1, 1, -1]),
        'groups': [3],
        'result': [0, 0, 1, 1, 1],
        'skip': False,
    },
    {
        'line': np.array([1, -1, -1, -1, -1, -1, 1]),
        'groups': [2, 2],
        'result': np.array([1, 1, 0, 0, 0, 1, 1]),
        'skip': False,
    },
    {
        'line': np.array([1, 0, 1, -1, -1, -1, -1, -1, 1, 0, 1]),
        'groups': [1, 2, 2, 1],
        'result': np.array([1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1]),
        'skip': False,
    },
    {
        'line': np.array([-1, -1, -1, 0, 1, -1, -1, -1, -1]),
        'groups': [2, 2],
        'result': np.array([-1, -1, -1, 0, 1, 1, 0, -1, -1]),
        'skip': False,
    },
    {
        'line': np.array([-1, -1, -1, -1, 0, -1, 1, -1, -1, -1, -1, -1, -1]),
        'groups': [3, 3],
        'result': np.array([-1, -1, -1, -1, 0, -1, 1, 1, -1, -1, -1, -1, -1]),
        'skip': True,
    },
]

SOLVE_LINE_IMPOSSIBLE_CASES = [
    {
        'line': np.array([-1, -1, 0, -1, -1]),
        'groups': [3],
        'result': [],
        'skip': False,
    },
    {
        'line': np.array([-1, -1, -1, -1]),
        'groups': [1, 1, 1],
        'result': [],
        'skip': False,
    },
]


def invert(test_case):
    return {
        'line': test_case['line'][::-1],
        'groups': test_case['groups'][::-1],
        'result': test_case['result'][::-1],
        'skip': False,
    }

def test_case_str(test_case):
    return (
        f'{test_case["groups"]} | '
        f'{line_state_str(test_case["line"], False)!r} -> {line_state_str(test_case["result"], False)!r}'
    )


class TestSolver(unittest.TestCase):
    def test_solve_line(self):
        for test_case in SOLVE_LINE_TEST_CASES:
            print(test_case_str(test_case))
            if test_case['skip']:
                print('(skipped)')
            else:
                improved = solve_line(test_case['line'].copy(), test_case['groups'])
                improved = solve_line(improved, test_case['groups'])

                np.testing.assert_equal(improved, test_case['result'])

                inverted_case = invert(test_case)
                print(test_case_str(inverted_case))
                improved = solve_line(inverted_case['line'].copy(), inverted_case['groups'])
                improved = solve_line(improved, inverted_case['groups'])

                np.testing.assert_equal(improved, inverted_case['result'])

    def test_impossible_cases(self):
        for test_case in SOLVE_LINE_IMPOSSIBLE_CASES:
            if not test_case['skip']:
                with self.assertRaises(AssertionError):
                    improved = solve_line(test_case['line'].copy(), test_case['groups'])
                    improved = solve_line(improved, test_case['groups'])


if __name__ == '__main__':
    unittest.main()
