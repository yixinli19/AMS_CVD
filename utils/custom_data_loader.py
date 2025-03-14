import pandas as pd
import scipy
from scipy.io import loadmat

from helper_code import find_all_challenge_files, get_frequency, get_labels, get_num_samples
from utils import load_header_with_fallback, butterworth_elgendi_rpeak
from scipy.signal import resample

import numpy as np
import matplotlib.pyplot as plt
from const import *




def extract_ecg_cycles_lead_one(recording, frequency, num_samples, target_length=256, cycle_num=5, overlap=3):
    lead = recording[0]  # Choose the first lead here

    if len(lead) != num_samples:
        return None, None

    r_peak_indices, filtered_signal = butterworth_elgendi_rpeak(lead, frequency)

    if r_peak_indices is None or len(r_peak_indices) < cycle_num + 1:
        return None, None

    cycles = []
    durations = []

    # Calculate the step size based on the overlap
    step = cycle_num - overlap
    if step <= 0:
        step = 1  # Ensure at least moving forward by one R-peak to avoid infinite loops

    for index in range(0, len(r_peak_indices) - cycle_num, step):
        # Extract the segment from the current R-peak to the R-peak at the end of the specified cycle number
        start_idx = r_peak_indices[index]
        end_idx = r_peak_indices[index + cycle_num]
        current_ecg = lead[start_idx:end_idx]

        # Resample the current ECG segment to have a uniform length
        current_ecg = resample(current_ecg, target_length * cycle_num)

        
        # t = np.arange(len(current_ecg)) / frequency  # Time vector in seconds

        # plt.figure(figsize=(12, 6))
        # plt.plot(t, current_ecg, color='blue', linewidth=1)
        # # plt.title('ECG Signal')
        # # plt.xlabel('Time (seconds)')
        # # plt.ylabel('Amplitude (mV)')
        # plt.grid(True)
        # plt.show()

        # Compute durations for each cycle in the segment
        cycle_durations = []
        for k in range(cycle_num):
            duration = (r_peak_indices[index + k + 1] - r_peak_indices[index + k]) / frequency
            cycle_durations.append(duration)

        cycles.append(current_ecg)
        durations.append(cycle_durations)

    return cycles, durations





import numpy as np
import scipy.io

def load_data_lead_one(paths, lead_num=1, max_circle=None, cycle_num=1, overlap=0):
    cnt = 0
    data_rows = []
    duration_list = []

    for path in paths:
        header_files, recording_files = find_all_challenge_files(path)
        length = len(header_files)

        for i in range(length):
            mat_file_path = recording_files[i]
            header_file_path = header_files[i]

            try:
                mat = loadmat(mat_file_path)
            except (scipy.io.matlab._miobase.MatReadError, FileNotFoundError):
                continue

            header = load_header_with_fallback(header_file_path)
            frequency = get_frequency(header)
            num_samples = get_num_samples(header)

            # Extract ECG cycles and individual cycle durations
            lead1_cycles, cycle_durations = extract_ecg_cycles(
                mat['val'], frequency, num_samples, lead_num, target_length=256, cycle_num=cycle_num, overlap=overlap
            )

            if lead1_cycles is None:
                continue

            diagnosis = get_labels(header)
            if len(diagnosis) == 1:
                diag_code = int(diagnosis[0])
                
                if diag_code in diagnosis_level_1:
                    diag_label = 1
                elif diag_code in diagnosis_level_2:
                    diag_label = 2
                elif diag_code in diagnosis_level_3:
                    diag_label = 3
                elif diag_code in diagnosis_level_4:
                    diag_label = 4
                elif diag_code in diagnosis_level_5:
                    diag_label = 5
                else:
                    continue

                cnt += 1

                # Prepare rows for each cycle
                for cycle, durations in zip(lead1_cycles, cycle_durations):
                    row_data = [diag_label] + list(cycle) + list(durations)
                    data_rows.append(row_data)

                    # Add individual cycle durations to the list
                    duration_list.extend(durations)

            if max_circle is not None and len(data_rows) >= max_circle:
                break

    # Convert the list of lists to a NumPy array for faster dataframe creation
    data_array = np.array(data_rows)

    # Create a DataFrame with column names
    num_points = len(lead1_cycles[0]) if lead1_cycles else 0
    num_durations = len(cycle_durations[0]) if cycle_durations else 0

    columns = ['diagnosis'] + [f'point_{i + 1}' for i in range(num_points)] + [f'cycle_duration_{i + 1}' for i in range(num_durations)]
    df = pd.DataFrame(data_array, columns=columns)

    print(f"Number of records: {cnt}")
    return df, duration_list


def extract_ecg_cycles(recording, frequency, num_samples, num_lead, target_length=256, cycle_num=5, overlap=3):
    all_cycles = []
    all_durations = []

    # Iterate through all leads
    for lead_index in range(num_lead):
        lead = recording[lead_index]

        if len(lead) != num_samples:
            # Append empty list if the lead length is not as expected
            all_cycles.append([])
            all_durations.append([])
            continue

        # Get R-peak indices and filtered signal for the current lead
        r_peak_indices, filtered_signal = butterworth_elgendi_rpeak(lead, frequency)

        if r_peak_indices is None or len(r_peak_indices) < cycle_num + 1:
            # Append empty list if no valid R-peaks are found
            all_cycles.append([])
            all_durations.append([])
            continue

        cycles = []
        durations = []

        # Calculate the step size based on the overlap
        step = cycle_num - overlap
        if step <= 0:
            step = 1  # Ensure at least moving forward by one R-peak to avoid infinite loops

        # Extract cycles based on R-peak indices
        for index in range(0, len(r_peak_indices) - cycle_num, step):
            start_idx = r_peak_indices[index]
            end_idx = r_peak_indices[index + cycle_num]
            current_ecg = lead[start_idx:end_idx]

            # Resample the current ECG segment to have a uniform length
            current_ecg = resample(current_ecg, target_length * cycle_num)

            # Compute durations for each cycle in the segment
            cycle_durations = []
            for k in range(cycle_num):
                duration = (r_peak_indices[index + k + 1] - r_peak_indices[index + k]) / frequency
                cycle_durations.append(duration)

            cycles.append(current_ecg)
            durations.append(cycle_durations)

        # Append the cycles and durations for the current lead
        all_cycles.append(cycles)
        all_durations.append(durations)

    return all_cycles, all_durations



def load_data(paths, lead_num=1, max_circle=None, cycle_num=1, overlap=0):
    cnt = 0
    data_rows = []
    duration_list = []

    # Calculate the number of columns based on lead, cycle, and target length
    target_length = 256
    num_points = target_length * cycle_num * lead_num
    num_durations = cycle_num * lead_num

    # Create an empty DataFrame with the required number of columns
    columns = ['diagnosis'] + [f'lead_point_{i + 1}' for i in range(num_points)] + [f'cycle_duration_{i + 1}' for i in range(num_durations)]
    df = pd.DataFrame(columns=columns)

    for path in paths:
        header_files, recording_files = find_all_challenge_files(path)
        length = len(header_files)

        for i in range(length):
            mat_file_path = recording_files[i]
            header_file_path = header_files[i]

            try:
                mat = loadmat(mat_file_path)
            except (scipy.io.matlab._miobase.MatReadError, FileNotFoundError):
                continue

            header = load_header_with_fallback(header_file_path)
            frequency = get_frequency(header)
            num_samples = get_num_samples(header)

            # Extract ECG cycles and individual cycle durations for all leads
            all_lead_cycles, all_cycle_durations = extract_ecg_cycles(
                mat['val'], frequency, num_samples, lead_num, target_length=target_length, cycle_num=cycle_num, overlap=overlap
            )


            # # Print the shape of all cycles and all durations
            # print(f"Shape of all_lead_cycles: {len(all_lead_cycles)} leads, {[len(lead) for lead in all_lead_cycles]} cycles per lead")
            # print(f"Shape of all_cycle_durations: {len(all_cycle_durations)} leads, {[len(lead) for lead in all_cycle_durations]} durations per lead")
            # if len(all_lead_cycles) > 0 and len(all_lead_cycles[0]) > 0:
            #     print(f"Number of data points per cycle: {len(all_lead_cycles[0][0])}")

            if all_lead_cycles is None:
                continue

            diagnosis = get_labels(header)
            if len(diagnosis) == 1:
                diag_code = int(diagnosis[0])
                
                if diag_code in diagnosis_level_1:
                    diag_label = 1
                elif diag_code in diagnosis_level_2:
                    diag_label = 2
                elif diag_code in diagnosis_level_3:
                    diag_label = 3
                elif diag_code in diagnosis_level_4:
                    diag_label = 4
                elif diag_code in diagnosis_level_5:
                    diag_label = 5
                else:
                    continue

                cnt += 1

                # Prepare rows for each cycle (one cycle per row)
                for cycle_index in range(len(all_lead_cycles[0])):
                    combined_cycles = []
                    combined_durations = []

                    # Flatten all lead data for the current cycle into a single row
                    for lead_index in range(len(all_lead_cycles)):
                        lead_cycles = all_lead_cycles[lead_index]
                        lead_durations = all_cycle_durations[lead_index]

                        if lead_cycles is None or cycle_index >= len(lead_cycles):
                            continue

                        combined_cycles.extend(lead_cycles[cycle_index])
                        combined_durations.extend(lead_durations[cycle_index])
                        
                        duration_list.extend(lead_durations[cycle_index])

                    # Ensure the row has the correct length by padding with zeros if necessary
                    if len(combined_cycles) < target_length * lead_num:
                        combined_cycles.extend([0] * (target_length * lead_num - len(combined_cycles)))
                    if len(combined_durations) < lead_num:
                        combined_durations.extend([0] * (lead_num - len(combined_durations)))

                    row_data = [diag_label] + combined_cycles + combined_durations
                    data_rows.append(row_data)

            if max_circle is not None and len(data_rows) >= max_circle:
                break

    # Convert the list of lists to a DataFrame
    df = pd.DataFrame(data_rows, columns=columns)

    print(f"Number of records: {cnt}")
    return df, duration_list
