import argparse

class CountPo:
    def __init__ (self, user_input):
        self.input = user_input
        self.x = self.input*10

    def counting(self):
        for i in range(self.input):
            print(i)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_1", type=int)
    parser.add_argument("--input_2", type=int)
    args = parser.parse_args()

    a = args.input_1
    b = args.input_2

    counter = count(a)
    # counter.counting()
    #
    # countr_2 = count(b)
    # countr_2.counting()

    print(counter.user_input)
    print(counter.x)

if __name__ == "__main__":

    main()