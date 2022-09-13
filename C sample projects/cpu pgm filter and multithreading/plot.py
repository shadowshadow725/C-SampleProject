# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages


def main():
    pdf = PdfPages('3x3 result.pdf')
    f = open("SEQUENTIAL.txt")
    d = f.read()

    a = d.split('\n')
    while '' in a:
        a.remove('')

    fig, ax = plt.subplots()
    plt.suptitle("Test results 3x3 filter")
    plt.ylabel('time(ms)')
    plt.xlabel('threads')
    tmp = float(a[1][0:-2])
    plt.hlines(float(a[0][0:-2]), 0, 8, 'r')
    red_patch = mpatches.Patch(color='red', label='sequential')

    # plt.hlines(float(a[1][0:-2]), 0, 8, 'b')
    f.close()
    f = open("SHARDED_ROWS.txt")
    d = f.read()
    b = []
    a = d.split('\n')
    for i in a:
        b.append(i.split(' '))


    row3 = b[0:8]
    row5 = b[9:17]
    row3_thread = []
    row3_time = []
    row5_thread = []
    row5_time = []
    for i in row3:
        row3_thread.append(float(i[1]))
        row3_time.append(float(i[0][0:-2]))
    for i in row5:
        row5_thread.append(float(i[1]))
        row5_time.append(float(i[0][0:-2]))
    plt.plot(row3_thread, row3_time, 'g-')
    green_patch = mpatches.Patch(color='green', label='sharded rows')


    f.close()
    f = open("SHARDED_COLUMNS_COLUMN_MAJOR.txt")
    d = f.read()
    b = []
    a = d.split('\n')
    for i in a:
        b.append(i.split(' '))
    cc3 = b[0:8]
    cc5 = b[9:17]
    cc3_thread = []
    cc3_time = []
    cc5_thread = []
    cc5_time = []
    for i in cc3:
        cc3_thread.append(float(i[1]))
        cc3_time.append(float(i[0][0:-2]))
    for i in cc5:
        cc5_thread.append(float(i[1]))
        cc5_time.append(float(i[0][0:-2]))
    plt.plot(cc3_thread, cc3_time, 'b-')
    blue_patch = mpatches.Patch(color='blue', label='sharded columns column major')

    f.close()

    f = open("SHARDED_COLUMNS_ROW_MAJOR.txt")
    d = f.read()
    b = []
    a = d.split('\n')
    for i in a:
        b.append(i.split(' '))
    cr3 = b[0:8]
    cr5 = b[9:17]
    cr3_thread = []
    cr3_time = []
    cr5_thread = []
    cr5_time = []
    for i in cr3:
        cr3_thread.append(float(i[1]))
        cr3_time.append(float(i[0][0:-2]))
    for i in cr5:
        cr5_thread.append(float(i[1]))
        cr5_time.append(float(i[0][0:-2]))
    plt.plot(cr3_thread, cr3_time, 'k-')
    black_patch = mpatches.Patch(color='black', label='sharded columns row major')
    plt.legend(handles=[red_patch, blue_patch, green_patch, black_patch])
    f.close()

    plt.savefig(pdf, format='pdf')
    pdf.close()


    plt.figure()
    pdf = PdfPages('5x5 result.pdf')
    plt.suptitle("Test results 5x5 filter")
    plt.ylabel('time(ms)')
    plt.xlabel('threads')
    plt.legend(handles=[red_patch, blue_patch, green_patch, black_patch])
    plt.hlines(tmp, 0, 8, 'r')
    plt.plot(row5_thread, row5_time, 'b-')
    plt.plot(cc5_thread, cc5_time, 'g-')
    plt.plot(cr5_thread, cr5_time, 'k-')
    plt.savefig(pdf, format='pdf')
    pdf.close()
    

    plt.figure()
    pdf = PdfPages('Work Queue test result.pdf')
    plt.suptitle("Work Queue Test results 3x3 filter")
    plt.ylabel('time(ms)')
    plt.xlabel('threads')
    f = open('WORK_QUEUE.txt')
    d = f.read()
    a = d.split('\n')
    b = []
    for i in a:
        b.append(i.split(' '))

    d1_thread, d2_thread, d3_thread, d4_thread, d5_thread = [], [], [], [], []
    d1_time, d2_time, d3_time, d4_time, d5_time = [], [], [], [], []
    d1 = b[0:8]
    d2 = b[9:17]
    d3 = b[18:26]
    d4 = b[27:35]
    d5 = b[36:44]

    for i in range(8):

        d1_thread.append(float(d1[i][1]))
        d1_time.append(float(d1[i][0][0:-2]))
        d2_thread.append(float(d2[i][1]))
        d2_time.append(float(d2[i][0][0:-2]))
        d3_thread.append(float(d3[i][1]))
        d3_time.append(float(d3[i][0][0:-2]))
        d4_thread.append(float(d4[i][1]))
        d4_time.append(float(d4[i][0][0:-2]))
        d5_thread.append(float(d5[i][1]))
        d5_time.append(float(d5[i][0][0:-2]))

    plt.plot(d1_thread, d1_time, 'r-')
    plt.plot(d2_thread, d2_time, 'g-')
    plt.plot(d3_thread, d3_time, 'b-')
    plt.plot(d4_thread, d4_time, 'k-')
    plt.plot(d5_thread, d5_time, 'c-')
    red_patch = mpatches.Patch(color='red', label='work chunk = ' + d1[0][2])
    green_patch = mpatches.Patch(color='green', label='work chunk = ' + d2[0][2])
    blue_patch = mpatches.Patch(color='blue', label='work chunk = ' + d3[0][2])
    black_patch = mpatches.Patch(color='black', label='work chunk = ' + d4[0][2])
    cyan_patch = mpatches.Patch(color='cyan', label='work chunk = ' + d5[0][2])
    plt.legend(handles=[red_patch,green_patch, blue_patch, black_patch, cyan_patch])
    plt.savefig(pdf, format='pdf')
 
    plt.figure()
    f.close()


    plt.suptitle("Work Queue Test results 5x5 filter")
    plt.ylabel('time(ms)')
    plt.xlabel('threads')
    d1_thread, d2_thread, d3_thread, d4_thread, d5_thread = [], [], [], [], []
    d1_time, d2_time, d3_time, d4_time, d5_time = [], [], [], [], []

    d1 = b[45:53]
    d2 = b[54:62]
    d3 = b[63:71]
    d4 = b[72:90]
    d5 = b[81:99]

    for i in range(8):
        d1_thread.append(float(d1[i][1]))
        d1_time.append(float(d1[i][0][0:-2]))
        d2_thread.append(float(d2[i][1]))
        d2_time.append(float(d2[i][0][0:-2]))
        d3_thread.append(float(d3[i][1]))
        d3_time.append(float(d3[i][0][0:-2]))
        d4_thread.append(float(d4[i][1]))
        d4_time.append(float(d4[i][0][0:-2]))
        d5_thread.append(float(d5[i][1]))
        d5_time.append(float(d5[i][0][0:-2]))

    plt.plot(d1_thread, d1_time, 'r-')
    plt.plot(d2_thread, d2_time, 'g-')
    plt.plot(d3_thread, d3_time, 'b-')
    plt.plot(d4_thread, d4_time, 'k-')
    plt.plot(d5_thread, d5_time, 'c-')
    red_patch = mpatches.Patch(color='red', label='work chunk = ' + d1[0][2])
    green_patch = mpatches.Patch(color='green', label='work chunk = ' + d2[0][2])
    blue_patch = mpatches.Patch(color='blue', label='work chunk = ' + d3[0][2])
    black_patch = mpatches.Patch(color='black', label='work chunk = ' + d4[0][2])
    cyan_patch = mpatches.Patch(color='cyan', label='work chunk = ' + d5[0][2])
    plt.legend(handles=[red_patch, green_patch, blue_patch, black_patch, cyan_patch])
    plt.savefig(pdf, format='pdf')
    pdf.close()











if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
