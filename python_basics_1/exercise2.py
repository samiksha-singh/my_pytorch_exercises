import argparse


class Counter:
    def __init__(self,user_input):
        self.user_input = user_input

    def add(self,my_input):
        self.user_input += my_input

    def get_final_value(self):
        return self.user_input

class Mul(Counter):
    def __init__(self,user_input):
        super().__init__(user_input)

    def multiply(self,in_value):
        self.user_input *= in_value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--init", type=int, required=True)
    args = parser.parse_args()
    init = args.init

    count = Mul(init)
    count.add(1)
    count.multiply(2)
    print(count.get_final_value())

if __name__ == "__main__":
    main()


