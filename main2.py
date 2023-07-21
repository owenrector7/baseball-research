import sys


# def gs():
#     preferences = {}
#     matches = {}


# def create_mentors(hos_count, hos_capcities, hos_prefs):
#     mentors = []

#     # take in a list of how many hospitals there are, their capacities, and their prefs
#     # for each hospital, look at its capacity in the capacity array, look at its prefs in the pref array, multiply the capac
#     # for each one in the capacity, make a mentor—these mentors will each have the same preferences as the hospital

#     # output a list of mentors, this will be a 2 dimensional list with their preferences

#     pass


def main():

    count = 0

    hos_prefs = []
    res_prefs = []
    h_capacity = []

    for line in sys.stdin:

        print(line)

#         if count == 0:
#             count_list = [int(elem) for elem in line.split()]
#             h_count = count_list[0]
#             r_count = count_list[1]
#             count += 1

#         if count in range(1, h_count+1):
#             capacity_list = [int(elem) for elem in line.split()]
#             h_capacity.append(capacity_list)
#             count += 1

#         if count in range(h_count+1, (2*h_count)+1):
#             pref_list1 = [int(elem) for elem in line.split()]
#             hos_prefs.append(pref_list1)
#             count += 1

#         if count in range((2*h_count)+1, (2*h_count)+(r_count+1)):
#             pref_list2 = [int(elem) for elem in line.split()]
#             res_prefs.append(pref_list2)
#             count += 1

#     # print(hos_prefs)
#     # print(res_prefs)
#     # print(h_capacity)
#     print(count)


main()
