""" script to move Slurm log files to their respective results directories """

from argparse import ArgumentParser
from datetime import datetime
from glob import glob
import os


# command-line arguments
parser = ArgumentParser(description='script to move Slurm log files to their '
                                    'respective results directories')
parser.add_argument('pattern', type=str, help='pattern of log file names')
parser.add_argument('--exp-file', type=str,
                    default='results/completed_experiments.txt',
                    help='path to file for keeping track of directories of '
                         'completed experiments')
parser.add_argument('--dry-run', action='store_true',
                    help='print actions without moving files')


def main(args):
    """ main function """

    # get all Slurm output files based on filename pattern
    files = sorted(glob(args.pattern))

    # load completed experiments and their completion times in dictionary
    experiments = dict()
    with open(args.exp_file, 'r') as f:
        for line in f.readlines():
            date, exp_name = line.strip().split('\t')
            experiments[exp_name] = date

    # iterate through Slurm output files
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    for fname in files:
        with open(fname, 'r') as f:
            lines = f.readlines()
        output_dir = None
        completed = False

        # iterate through lines in Slurm output file
        for line in lines:
            if 'python' in line and 'output_dir' in line:
                # get output directory from line with Python command
                options = line.strip().split(' ')
                options = [elem for elem in options if 'output_dir' in elem]
                _, output_dir = options[0].split('=')
            elif 'predict metrics' in line:
                # indicates that experiment completed
                completed = True

        # handle all cases (output directory not found in log file or does not
        # exist, experiment not completed, or experiment is valid + completed)
        if output_dir is None:
            print(f'no output_dir for {fname}')
        else:
            if not os.path.exists(output_dir):
                print('output_dir does not exist')
            elif not completed:
                print(f'{fname} not completed')
            else:
                print(f'moving {fname} to {output_dir}')
                if not args.dry_run:
                    newname = os.path.join(output_dir, 'slurm_output.log')
                    os.rename(fname, newname)
                    experiments[os.path.dirname(output_dir)] = now_str

        # save experiments and their completion times to file
        with open(args.exp_file, 'w') as f:
            exp_names = list(experiments.keys())
            for exp_name in sorted(exp_names, key=lambda x: experiments[x]):
                f.write(f'{experiments[exp_name]}\t{exp_name}\n')


if __name__ == '__main__':
    main(parser.parse_args())

