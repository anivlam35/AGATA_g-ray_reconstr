import numpy as np, pandas as pd
from lib import config


def cutting_off(fname, out_filename):
    f1 = open(fname, "r")
    raw_data = ('\n').join(f1.read().split('\n')[2316:])
    f1.close()

    title = "event_num,interaction_num,crystal,edep,x,y,z,slice_sect,time\n"
    open(f'scrapped_data_{out_filename}.csv', 'w').close()
    scrapped_df = open(f'scrapped_data_{out_filename}.csv', 'a')
    scrapped_df.write(title)

    raw_data = raw_data.split('   -1   ')[1:]

    event_number = 0
    for event in raw_data:
        event_number += 1
        rows = event.split('\n')[1:-1]
        interaction_num = 1
        for row in rows:
            if row[0] == '-':
                if rows.index(row) == 0:
                    event_number -= 1
                break
            else:
                formated_row = ','.join((str(event_number) + ',' + str(interaction_num) + row).split()) + '\n'
                scrapped_df.write(formated_row)
                interaction_num += 1


def main():
    cutting_off("OnlyGammaEvents.0001", 'mul1_big')


if __name__ == "__main__":
    main()
