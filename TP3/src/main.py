
from argparse import ArgumentParser


def initialize_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    # Required positional argument
    parser.add_argument('run_part', type=str, nargs='?', choices=['part_1', 'part_2'],
                        help='The part from the asignment to execute')
    # Optional positional argument
    # parser.add_argument('opt_pos_arg', type=int, nargs='?', help='An optional integer positional argument')
    # Optional argument
    # parser.add_argument('--opt_arg', type=int, help='An optional integer argument')
    parser.add_argument('--show_plots', action='store_true', default=False, help='A boolean switch')

    return parser

if __name__ == '__main__':
    parser = initialize_arg_parser()

    args = parser.parse_args()
    
    
