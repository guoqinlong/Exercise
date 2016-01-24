# coding=u8
import multiprocessing
import string
import itertools
import collections


class SimpleMapReduce:
    def __init__(self, map_func, reduce_func, num_of_workers=None):
        self._map_func = map_func
        self._reduce_func = reduce_func
        self.pool = multiprocessing.Pool(num_of_workers)

    @staticmethod
    def partition(mapped_values):
        partitioned_data = collections.defaultdict(list)
        for word, occurance in mapped_values:
            partitioned_data[word].append(occurance)

        return partitioned_data.items()

    def __call__(self, inputs, chunk_size=1):
        map_response = self.pool.map(self._map_func, inputs, chunk_size)
        partitioned_data = self.partition(itertools.chain(*map_response))
        reduced_values = self.pool.map(self._reduce_func, partitioned_data, chunk_size)
        return reduced_values


def file_to_words(filename):
    stop_words = {"a", "an", "and", "are"}

    tr = string.maketrans(string.punctuation, ' ' * len(string.punctuation))

    print multiprocessing.current_process().name, 'reading', filename

    output = []
    with open(filename, 'rt') as f:
        for line in f:
            if line.lstrip().startswith(".."):
                continue
            line = line.translate(tr)
            for word in line.split():
                word = word.lower()
                if word.isalpha() and word not in stop_words:
                    output.append((word, 1))
    return output


def count_words(item):
    word, occurance = item
    return word, sum(occurance)


if __name__ == '__main__':
    import glob
    import operator

    input_files = glob.glob('*.rst')
    mapper = SimpleMapReduce(file_to_words, count_words)
    word_counts = mapper(input_files)
    word_counts.sort(key=operator.itemgetter(1),
                     reverse=True)

    print "\nTop 20 Word Counts\n"

    top20 = word_counts[:20]
    max_length = max(len(word) for word in top20)
    for key, value in top20:
        print "%-*s : %5s" % (max_length + 1, key, value)
