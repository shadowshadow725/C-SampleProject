

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def print_hi(name):
    
    print(f'Hi, {name}') 


def main():
    f = open("data.txt")
    d = f.read()

    a = d.split('\n')
    b = []
    for i in a:
        b.append(i.split(' '))

    x = []
    y = []
    unique_set = set()
    unique_time = set()
    for i in b:
        if len(i) > 1:

            x.append(float(i[1]))
            y.append(float(i[0][0:-2]))
            if y[-1] not in unique_time:
                unique_set.add((x[-1], float(i[0][0:-2])))
                unique_time.add(float(i[0][0:-2]))
    x.pop(0)
    y.pop(0)

    
    with PdfPages('part1b.pdf') as pdf:
        fig, ax = plt.subplots()
        plt.plot(x, y, 'b-')
        plt.ylabel('latency ns')
        plt.xlabel('size written 2^x')
        
        plt.xticks(x)
        ax.tick_params(axis='x', rotation=90)
        pdf.savefig()
        plt.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
