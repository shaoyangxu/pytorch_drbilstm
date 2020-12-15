from argparse import ArgumentParser
def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--data_dir',default="/home/ywzhang/xsy/data/snli_1.0")
    parser.add_argument('--target_dir', default="/home/ywzhang/xsy/data/snli_1.0")
    parser.add_argument('--lowercase', default=False)
    parser.add_argument('--ignore_punctuation', default=False)
    parser.add_argument('--num_words', default=None)
    parser.add_argument('--labeldict', default={"entailment": 0,"neutral": 1,"contradiction": 2})
    parser.add_argument('--model', choices = ['bert', 'spanbert', 'roberta', 'xlnet'],default="bert")
    parser.add_argument('--model_size', choices = ['base', 'large'], default="base")
    parser.add_argument('--fine_tune', default=False)
    parser.add_argument('--aug_rate', default=0)

    return parser.parse_args()
