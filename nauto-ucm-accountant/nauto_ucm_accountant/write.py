"""
This script updates UCM accountant data
"""
from nauto_ucm_accountant.accountant import AccountantWriterS3
from nauto_ucm_accountant.logger import logger
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Update UCM accountant data')
    parser.add_argument(
        '--days',
        default=3,
        type=int,
        help='How many days backwards to update. Pass 0 to update everything'
    )
    return parser.parse_args()


def main(args):
    #TODO consider adding s3 path as commandline argument for the wrapper
    writer = AccountantWriterS3()
    if args.days == 0:
        logger.info("Updating all data since the beginning of the history")
        writer.write(last_days=0)
    elif args.days > 0:
        logger.info(f"Updating data for last {args.days} days")
        writer.write(last_days=args.days)
    else:
        logger.error(f"Invalid number of days: {args.days}")
        exit(1)


if __name__ == "__main__":
    main(parse_args())
