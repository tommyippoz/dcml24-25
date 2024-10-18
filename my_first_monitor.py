import csv
from time import sleep
import psutil

# Library for getting indicators -> https://psutil.readthedocs.io/en/latest/

def read_data():
    """
    This function reads data from the system
    :return: a dictionary
    """
    cpu_t = psutil.cpu_times()
    cpu_dict = {"idle_time": cpu_t.idle, "usr_time": cpu_t.user}
    cpu_dict["interrupt_time"] = cpu_t.interrupt
    return cpu_dict

def write_dict_to_csv(filename, dict_item, is_first_time):
    """
    This function writes a dictionary as a row of a CSV file
    :param filename:
    :param dict_item:
    :param is_first_time:
    :return:
    """
    if is_first_time:
        f = open(filename, 'w', newline="")
    else:
        f = open(filename, 'a', newline="")
    w = csv.DictWriter(f, dict_item.keys())
    if is_first_time:
        w.writeheader()
    w.writerow(dict_item)
    f.close()

if __name__ == "__main__":
    """
    Main of the monitor
    """

    is_first_time = True
    while True:
        cpu_dict = read_cpu_usage()
        write_dict_to_csv("my_first_dataset.csv", cpu_dict, is_first_time)
        is_first_time = False
        print(cpu_dict)
        sleep(1)