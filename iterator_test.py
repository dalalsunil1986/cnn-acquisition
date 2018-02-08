class FooIterator:
    def __init__(self):
        self.idx = 0
        self.length = 1000

    # Tell the world we are iteratable, and the thing that has the __next__() function available
    # is in fact ourselves
    def __iter__(self):
        return self


    def calc_fibbonaci(self, val):
        if val == 0:
            return 0
        if val == 1:
            return 1

        return self.calc_fibbonaci(val - 1) + self.calc_fibbonaci(val - 2)

    # This is the function that gets called within the for loop
    # to get the next thing.
    def __next__(self):
        # Raise StopIteration() when we're actually done doing stuff
        if self.idx > self.length:
            raise StopIteration()

        self.idx += 1

        # Return the current value
        return self.calc_fibbonaci(self.idx)

    # Return our self-imposed length
    def __len__(self):
        return self.length

fi = FooIterator()

for val in fi:
    print(val)