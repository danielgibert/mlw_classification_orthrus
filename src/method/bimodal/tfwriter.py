import argparse
import tensorflow as tf
import os
project_path = os.path.dirname(os.path.realpath("../../../"))
import sys
import csv
sys.path.append(project_path)
from metaphor.metaphor_engine import MetaPHOR
from src.method.utils import load_vocabulary, serialize_bimodal_example

def dataset_to_tfrecords(pe_filepath,
                         tfrecords_filepath,
                         labels_filepath,
                         opcodes_vocabulary_mapping_filepath,
                         bytes_vocabulary_mapping_filepath,
                         max_mnemonics=50000,
                         max_bytes=2000000):

    opcodes_vocabulary_mapping = load_vocabulary(opcodes_vocabulary_mapping_filepath)
    bytes_vocabulary_mapping = load_vocabulary(bytes_vocabulary_mapping_filepath)
    tfwriter = tf.io.TFRecordWriter(tfrecords_filepath)

    i = 0

    # Training TFRecord
    with open(labels_filepath, "r") as labels_file:
        reader = csv.DictReader(labels_file, fieldnames=["Id",
                                                           "Class"])
        reader.__next__()
        for row in reader:
            print("{};{}".format(i, row['Id']))
            metaPHOR = MetaPHOR(pe_filepath + row['Id'] + ".asm")

            # Extract opcodes
            opcodes = metaPHOR.get_opcodes_data_as_list(opcodes_vocabulary_mapping)

            if len(opcodes) < max_mnemonics:
                while len(opcodes) < max_mnemonics:
                    opcodes.append("PAD")
            else:
                opcodes = opcodes[:max_mnemonics]
            raw_mnemonics = " ".join(opcodes)

            # Extract bytes
            bytes_sequence = metaPHOR.get_hexadecimal_data_as_list()
            for i in range(len(bytes_sequence)):
                if bytes_sequence[i] not in bytes_vocabulary_mapping.keys():
                    bytes_sequence[i] = "UNK"

            if len(bytes_sequence) < max_bytes:
                while len(bytes_sequence) < max_bytes:
                    bytes_sequence.append("PAD")
            else:
                bytes_sequence = bytes_sequence[:max_bytes]
            raw_bytes = " ".join(bytes_sequence)

            example = serialize_bimodal_example(raw_mnemonics, raw_bytes, int(row['Class'])-1)
            tfwriter.write(example)
            i += 1