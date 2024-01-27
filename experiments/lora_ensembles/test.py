import argparse
if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--member_id', type=int, required=True)
    args = parser.parse_args()

    # Now you can use args.epoch and args.member_id in your script
    epoch = args.epoch
    member_id = args.member_id
    print("Epoch ", epoch)
    print("Member_id ", member_id)