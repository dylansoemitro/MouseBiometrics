import numpy as np
import pandas as pd
import glob
import random
import time
import pickle
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
from pathlib import Path
import os
import sys
from sklearn.model_selection import train_test_split
#get preprocessed data

def get_preprocessed_data(dataset):
    path = "Data/PrepedCSVs/" + dataset + ".csv"
    df = pd.read_csv(path)
    return df

#window generator function
def time_window(dataframe, delta_time, shift=None, drop_remainder=False):
    if len(dataframe) == 0:
        return
    p0 = dataframe.index[0]
    while True:
        p = p0
        while dataframe["time"][p] < delta_time + dataframe["time"][p0]:
            p += 1
            if p == dataframe.index[-1]:
                if not drop_remainder:
                    yield dataframe.loc[p0:]
                return

        yield dataframe.loc[p0:p]
        if shift is None:
            p0 = p
        else:
            while dataframe["time"][p0] < shift + dataframe["time"][p]:
                p0 += 1


## Feature Functions

# returns the total time in seconds between the interval or the delta times between each mouse move
def get_elapsed_time(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    return user_data["time"][end] - user_data["time"][start]


# returns distance in the x direction
def get_x_distance(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    return user_data["px"][end] - user_data["px"][start]


# returns distance in the y direction
def get_y_distance(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    return user_data["py"][end] - user_data["py"][start]


# returns distance
def get_euclidean_distance(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    xdist = get_x_distance(user_data, start, end)
    ydist = get_y_distance(user_data, start, end)
    edist = np.sqrt(np.square(xdist) + np.square(ydist))
    return edist


def get_manhattan_distance(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    return get_x_distance(user_data, start, end) + get_y_distance(user_data, start, end)


# returns velocity in the x direction
def get_x_velocity(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    if get_elapsed_time(user_data, start, end) == 0:
        return 0
    else:
        return get_x_distance(user_data, start, end) / get_elapsed_time(user_data, start, end)


# returns velocity in the y direction
def get_y_velocity(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    if get_elapsed_time(user_data, start, end) == 0:
        return 0
    else:
        return get_y_distance(user_data, start, end) / get_elapsed_time(user_data, start, end)


# return speed
def get_speed(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    if get_elapsed_time(user_data, start, end) == 0:
        return 0
    else:
        return get_euclidean_distance(user_data, start, end) / get_elapsed_time(user_data, start, end)


# TODO: verify accuracy
# return angular velocity
def get_angular_velocity(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    time = get_elapsed_time(user_data, start, end)
    point1 = (user_data["px"][start], user_data["py"][start])
    point2 = (user_data["px"][end], user_data["py"][end])
    ang1 = np.arctan2(*point1[::-1])
    ang2 = np.arctan2(*point2[::-1])
    ang_between = np.rad2deg((ang1 - ang2) % (2 * np.pi))
    if time == 0:
        return 0
    else:
        return ang_between / time


# returns linear acceleration
def get_acceleration(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    if end - start < 3:
        return 0
    delta_time = get_elapsed_time(user_data, start, end)
    start_speed = get_speed(user_data, start, start + 1)
    end_speed = get_speed(user_data, end - 1, end)
    if delta_time == 0:
        return 0
    else:
        return (end_speed - start_speed) / delta_time


def get_jerk(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    if end - start < 4:
        return 0
    delta_time = get_elapsed_time(user_data, start, end)
    start_acceleration = get_acceleration(user_data, start, start + 3)
    end_acceleration = get_acceleration(user_data, end - 3, end)
    if delta_time == 0:
        return 0
    else:
        jerk = (end_acceleration - start_acceleration) / delta_time
        return jerk


# TODO: verify accuracy
# returns curvature
def get_curvature(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    dist = get_euclidean_distance(user_data, start, end)
    point1 = (user_data["px"][start], user_data["py"][start])
    point2 = (user_data["px"][end], user_data["py"][end])
    ang1 = np.arctan2(*point1[::-1])
    ang2 = np.arctan2(*point2[::-1])
    ang_between = np.rad2deg((ang1 - ang2) % (2 * np.pi))

    curv = ang_between / dist
    if np.isnan(curv) or np.isinf(curv):
        return 0
    return curv


def get_curvature_change(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    if end - start < 3:
        return 0
    start_curv = get_curvature(user_data, start, start + 1)
    end_curv = get_curvature(user_data, end - 1, end)
    dist = get_euclidean_distance(user_data, start, end)

    dcurv = (end_curv - start_curv) / dist
    if np.isnan(dcurv) or np.isinf(dcurv):
        return 0
    return dcurv


def get_critical_points(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    if abs(get_curvature(user_data, start, end)) > np.pi / 10:
        return 1
    else:
        return 0


def get_direction(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    ydist = get_y_distance(user_data, start, end)
    if ydist == 0:
        return 0
    xdist = get_x_distance(user_data, start, end)
    if xdist == 0:
        return np.pi / 2
    return np.arctan(ydist / xdist)


def get_angle_of_curvature(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    d1 = get_euclidean_distance(user_data, start + 1, start)
    d3 = get_euclidean_distance(user_data, start + 1, start + 2)
    d2 = get_euclidean_distance(user_data, start, start + 2)
    numerator = np.square(d1) + np.square(d2) - np.square(d3)

    denominator = 2 * get_euclidean_distance(user_data, start + 1, start) * get_euclidean_distance(user_data, start + 1,
                                                                                                   start + 2)
    if denominator == 0:
        return 0
    else:
        return np.arccos(numerator / denominator)


def get_curvature_distance(user_data, start, end):
    if start + 2 > user_data.index[-1]:
        return 0

    dx = get_x_distance(user_data, start, start + 2)
    dy = get_x_distance(user_data, start, start + 2)
    numerator = dy * user_data["px"][start + 1] + dx * user_data["py"][start + 1] + (
                user_data["px"][start] * user_data["py"][start + 2] - user_data["px"][start + 2] * user_data["py"][
            start])
    distance_xy_2 = np.sqrt(np.square(dx) + np.square(dy))
    if distance_xy_2 == 0:
        return 0
    else:
        return numerator / distance_xy_2


def get_angle(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    if end - start < 3:
        return 0
    numerator = np.square(get_euclidean_distance(user_data, start, user_data.index[0])) + np.square(
        get_euclidean_distance(user_data, start, user_data.index[-1])) - np.square(
        get_euclidean_distance(user_data, user_data.index[0], user_data.index[-1]))
    denominator = 2 * get_euclidean_distance(user_data, user_data.index[0], start) * get_euclidean_distance(user_data,
                                                                                                            start,
                                                                                                            user_data.index[
                                                                                                                -1])
    if denominator == 0 or numerator / denominator < -1 or numerator / denominator > 0:
        return 0
    else:
        return np.arccos(numerator / denominator)


# not done
def get_curve_length_ratio(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    return get_euclidean_distance(user_data, start, end)


def get_straightness(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    dist = 0
    tot_dist = get_euclidean_distance(user_data, user_data.index[0], user_data.index[-1])
    for i in user_data.index[:-1]:
        dist += get_euclidean_distance(user_data, i, i + 1)
    if dist == 0:
        return 0
    return tot_dist / dist


def get_trajectory_center_of_mass(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    dist = 0
    for i in user_data.index[:-1]:
        dist += get_euclidean_distance(user_data, i, i + 1)
    elapsed_time = get_elapsed_time(user_data, start, end)
    current_distance = get_euclidean_distance(user_data, start, end)
    if dist == 0:
        return 0
    else:
        return (current_distance * elapsed_time) / dist


def get_scattering_coefficient(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    dist = 0
    for i in user_data.index[:-1]:
        dist += get_euclidean_distance(user_data, i, i + 1)
    TCM = 0
    for j in user_data.index[:-1]:
        TCM += get_trajectory_center_of_mass(user_data, start, end)
    elapsed_time = get_elapsed_time(user_data, start, end)
    current_distance = get_euclidean_distance(user_data, start, end)
    if dist == 0:
        return 0
    else:
        return (current_distance * np.square(elapsed_time) - np.square(TCM)) / dist


def get_third_moment(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    dist = 0
    for i in user_data.index[:-1]:
        dist += get_euclidean_distance(user_data, i, i + 1)
    elapsed_time = get_elapsed_time(user_data, start, end)
    current_distance = get_euclidean_distance(user_data, start, end)
    if dist == 0:
        return 0
    else:
        return (current_distance * (elapsed_time ** 3)) / dist


def get_fourth_moment(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    dist = 0
    for i in user_data.index[:-1]:
        dist += get_euclidean_distance(user_data, i, i + 1)
    elapsed_time = get_elapsed_time(user_data, start, end)
    current_distance = get_euclidean_distance(user_data, start, end)
    if dist == 0:
        return 0
    else:
        return (current_distance * (elapsed_time ** 4)) / dist


# ??????????????
def get_trajectory_curvature(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    dist = 0
    for i in user_data.index[:-1]:
        dist += get_euclidean_distance(user_data, i, i + 1)
    elapsed_time = get_elapsed_time(user_data, start, end)
    current_distance = get_euclidean_distance(user_data, start, end)
    return (current_distance * (elapsed_time ** 3)) / dist


def get_deviation(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    dx = get_x_distance(user_data, user_data.index[0], user_data.index[-1])
    dy = get_y_distance(user_data, user_data.index[0], user_data.index[-1])

    numerator = dy * user_data["px"][start] + dx * user_data["py"][start] + (
                user_data["px"][user_data.index[0]] * user_data["py"][user_data.index[1]] - user_data["px"][
            user_data.index[1]] * user_data["py"][user_data.index[0]])
    distance_xy = np.sqrt(np.square(dx) + np.square(dy))
    if distance_xy == 0:
        return 0
    else:
        return numerator / distance_xy


def get_velocity_curvature(user_data, start, end):
    if end > user_data.index[-1]:
        return 0
    jerk = get_jerk(user_data, start, end)
    acceleration = get_acceleration(user_data, start, end)
    if acceleration == 0:
        return 0
    else:
        return jerk / ((1 + acceleration ** 2) ** (3 / 2))


# wrapper for feature functions
class feature:
    def __init__(self, feature_func, return_func, offset=1):
        self.feature_func = feature_func
        self.return_func = return_func
        self.offset = offset

    def get_feature(self, user_data):
        flist = []
        if self.offset is None:
            flist.append(self.feature_func(user_data, user_data.index[0], user_data.index[-1]))
        else:
            for i in user_data.index:
                f = self.feature_func(user_data, i, i + self.offset)
                if np.isnan(f):
                    continue
                flist.append(f)

        return self.return_func(flist)


#custom return functions

def nz_max(input_array):
    new_array = [i for i in input_array if i != 0]
    if len(new_array) == 0:
        return 0
    else:
        return np.max(new_array)


def nz_min(input_array):
    new_array = [i for i in input_array if i != 0]
    if len(new_array) == 0:
        return 0
    else:
        return np.min(new_array)


def nz_range(input_array):
    return nz_max(input_array) - nz_min(input_array)


def nz_std(input_array):
    new_array = [i for i in input_array if i != 0]
    if len(new_array) == 0:
        return 0
    else:
        return np.std(new_array)


def nz_mean(input_array):
    new_array = [i for i in input_array if i != 0]
    if len(new_array) == 0:
        return 0
    else:
        return np.mean(new_array)


def extract_features(train_path, test_path, data, user_ids, feature_functions, test_size=0.3, min_data=2000, window_length=5000):

    window_offset = None  # offset_time

    cols = list(["user_id"]) + list(feature_functions.keys())
    train_df = pd.DataFrame(columns=cols)
    test_df = pd.DataFrame(columns=cols)
    res = "y"
    if (os.path.exists(train_path) or os.path.exists(test_path)):
        res = input("Do you want to overwrite feature data?\n[y][n]: ")
    if res == "y":
        for user in user_ids:
            if len(train_df.index) == 0 and os.path.exists(train_path):
                exist = False
                with open(train_path, 'rt') as f:
                    s = csv.reader(f, delimiter=",")
                    for row in s:
                        if str(user) in row[0]:
                            exist = True
                            continue
                if exist == True:
                    continue
            if user != 0.0:
                user_data = data[data["user_id"] == user]
                if len(user_data.index) < min_data:
                    print("Skipping user: {}".format(user), len(user_data.index), len(user_data))
                    continue
                print("Extracting features from user: {}".format(user))
                # init feature dict
                user_feature = {key: [] for key in cols}

                # extract_features
                for window in time_window(user_data, window_length, window_offset):
                    user_feature["user_id"].append(user)
                    for key in feature_functions:
                        f = feature_functions[key].get_feature(window)
                        if np.isnan([f]):
                            print(key, f)
                        user_feature[key].append(f)
                # clear_output()

                # split training and testing
                user_train, user_test = train_test_split(pd.DataFrame.from_dict(user_feature), test_size=0.3,
                                                         shuffle=False)
                train_df = pd.concat([train_df, user_train])
                test_df = pd.concat([test_df, user_test])
                if user == user_ids[0]:
                    train_df.to_csv(train_path, index=False)
                    test_df.to_csv(test_path, index=False)
                else:
                    train_df = train_df.reset_index(drop=True)
                    test_df = test_df.reset_index(drop=True)

                    with open(train_path, 'a') as f:
                        train_df.to_csv(f, header=False, index=False)
                    with open(test_path, 'a') as f:
                        test_df.to_csv(f, header=False, index=False)

    else:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

    return train_df, test_df


feature_functions = {
    "elapsed_time": feature(get_elapsed_time, np.sum),
    "critical_points": feature(get_critical_points, np.sum),
    "stroke-length": feature(get_euclidean_distance, np.sum),
    "straightness": feature(get_straightness, nz_max),
    "trajectory_center_of_mass": feature(get_trajectory_center_of_mass, np.sum),
    # "scattering_coefficient": feature(get_scattering_coefficient, np.sum),
    "third_moment": feature(get_third_moment, np.sum),
    "fourth_moment": feature(get_fourth_moment, np.sum),
    "velocity_curvature": feature(get_velocity_curvature, nz_mean, 4),
    "xvelocity-mean": feature(get_x_velocity, nz_mean),
    "xvelocity-maximum": feature(get_x_velocity, nz_max),
    "xvelocity-minimum": feature(get_x_velocity, nz_min),
    "xvelocity-std": feature(get_x_velocity, nz_std),
    "xvelocity-range": feature(get_x_velocity, nz_range),
    "yvelocity-mean": feature(get_y_velocity, nz_mean),
    "yvelocity-maximum": feature(get_y_velocity, nz_max),
    "yvelocity-minimum": feature(get_y_velocity, nz_min),
    "yvelocity-std": feature(get_y_velocity, nz_std),
    "yvelocity-range": feature(get_y_velocity, nz_range),
    "tangential-velocity-mean": feature(get_speed, nz_mean),
    "tangential-velocity-maximum": feature(get_speed, nz_max),
    "tangential-velocity-minimum": feature(get_speed, nz_min),
    "tangential-velocity-std": feature(get_speed, nz_std),
    "tangential-velocity-range": feature(get_speed, nz_range),
    "acceleration-mean": feature(get_acceleration, nz_mean, 3),
    "acceleration-maximum": feature(get_acceleration, nz_max, 3),
    "acceleration-minimum": feature(get_acceleration, nz_min, 3),
    "acceleration-std": feature(get_acceleration, nz_std, 3),
    "acceleration-range": feature(get_acceleration, nz_range, 3),
    "jerk-mean": feature(get_jerk, nz_mean, 4),
    "jerk-maximum": feature(get_jerk, nz_max, 4),
    "jerk-minimum": feature(get_jerk, nz_min, 4),
    "jerk-std": feature(get_jerk, nz_std, 4),
    "jerk-range": feature(get_jerk, nz_range, 4),
    "angular_velocity-mean": feature(get_angular_velocity, nz_mean),
    "angular_velocity-maximum": feature(get_angular_velocity, nz_max),
    "angular_velocity-minimum": feature(get_angular_velocity, nz_min),
    "angular_velocity-std": feature(get_angular_velocity, nz_std),
    "angular_velocity-range": feature(get_angular_velocity, nz_range),
    "curvature-mean": feature(get_curvature, nz_mean),
    "curvature-maximum": feature(get_curvature, nz_max),
    "curvature-minimum": feature(get_curvature, nz_min),
    "curvature-std": feature(get_curvature, nz_std),
    "curvature-range": feature(get_curvature, nz_range),
    "curvature_change-mean": feature(get_curvature_change, nz_mean, 3),
    "curvature_change-maximum": feature(get_curvature_change, nz_max, 3),
    "curvature_change-minimum": feature(get_curvature_change, nz_min, 3),
    "curvature_change-std": feature(get_curvature_change, nz_std, 3),
    "curvature_change-range": feature(get_curvature_change, nz_range, 3),
    "direction-mean": feature(get_direction, nz_mean),
    "direction-maximum": feature(get_direction, nz_max),
    "direction-minimum": feature(get_direction, nz_min),
    "direction-std": feature(get_direction, nz_std),
    "direction-range": feature(get_direction, nz_range),
    "angle-mean": feature(get_angle, nz_mean, 3),
    "angle-maximum": feature(get_angle, nz_max, 3),
    "angle-minimum": feature(get_angle, nz_min, 3),
    "angle-std": feature(get_angle, nz_std, 3),
    "angle-range": feature(get_angle, nz_range, 3),
    "curvature_distance-mean": feature(get_curvature_distance, nz_mean),
    "curvature_distance-maximum": feature(get_curvature_distance, nz_max),
    "curvature_distance-minimum": feature(get_curvature_distance, nz_min),
    "curvature_distance-std": feature(get_curvature_distance, nz_std),
    "curvature_distance-range": feature(get_curvature_distance, nz_range),
    "deviation-mean": feature(get_deviation, nz_mean),
    "deviation-maximum": feature(get_deviation, nz_max),
    "deviation-minimum": feature(get_deviation, nz_min),
    "deviation-std": feature(get_deviation, nz_std),
    "deviation-range": feature(get_deviation, nz_range)}

def import_data(dataset):
    path = "Data/PrepedCSVs/" + dataset
    df = pd.read_csv(path)
    return df

def extract_features_from_df(dataset, feature_functions):
    df = import_data(dataset)
    return extract_features(
             "Data/Features/Training_"+dataset+"_Features.csv",
             "Data/Features/Testing_"+dataset+"_Features.csv",
             df, df["user_id"].unique(),
             feature_functions, 0.3)
if __name__ == "__main__":
    dataset = sys.argv
    extract_features_from_df(dataset, feature_functions)