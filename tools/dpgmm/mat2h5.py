import h5features as h5f
import argparse

def convert(mat_file, output_file):
    with h5f.Converter(output_file) as converter:
        converter.convert(mat_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mat_file')
    parser.add_argument('output_file')
    args = parser.parse_args()
    
    convert(args.mat_file, args.output_file)
