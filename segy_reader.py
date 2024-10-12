# alaaraji_segi/segy_reader.py

import segyio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import Stream, Trace, UTCDateTime
from obspy.clients.iris import Client
from scipy.signal import welch


def read_segy(file_path, n_samples=None):
    """
    Open a SEG-Y file using segyio.

    Args:
        file_path (str): Path to the SEG-Y file.
        n_samples (int, optional): Number of samples to read from each trace.
                                    If None, all samples will be read.

    Returns:
        np.ndarray: An array containing the trace data, or None if an error occurs.
    """
    try:
        with segyio.open(file_path, ignore_geometry=True) as segy_file:
            n_traces = segy_file.tracecount  # Number of traces
            total_samples = segy_file.samples.size  # Total number of samples

            # Set the number of samples to read
            if n_samples is None or n_samples > total_samples:
                n_samples = total_samples  # Read all samples if not specified

            # Initialize an array to store the data
            data_array = np.zeros((n_traces, n_samples), dtype=np.float32)

            # Read the data from each trace
            for i in range(n_traces):
                data_array[i, :] = segy_file.trace[i][:n_samples]

            print(f"Number of traces: {n_traces}")
            print(f"Number of samples per trace: {n_samples}")

            return data_array

    except FileNotFoundError:
        print(f"Error: The specified SEG-Y file '{file_path}' was not found.")
        return None
    except segyio.SegyIOError as e:
        print("Error reading SEG-Y file:", e)
        return None
    except Exception as e:
        print("An unexpected error occurred:", e)
        return None


def inspect_segy(file_path):
    """
    Inspect a SEG-Y file to extract metadata.

    Args:
        file_path (str): Path to the SEG-Y file.

    Returns:
        dict: A dictionary containing metadata information, or None if an error occurs.
    """
    metadata = {}
    try:
        with segyio.open(file_path, ignore_geometry=True) as segy_file:
            metadata['number_of_traces'] = segy_file.tracecount
            metadata['number_of_samples'] = segy_file.samples.size
            metadata['sample_interval'] = segy_file.header[0][segyio.TraceField.SAMPLE_INTERVAL]  # Sample interval
            metadata['format'] = segy_file.format  # File format

        return metadata
    except FileNotFoundError:
        print(f"Error: The specified SEG-Y file '{file_path}' was not found.")
        return None
    except segyio.SegyIOError as e:
        print("Error reading SEG-Y file:", e)
        return None
    except Exception as e:
        print("An unexpected error occurred:", e)
        return None


def get_event_info(event_id):
    """
    Retrieve information about a seismic event using obspy.

    Args:
        event_id (str): The ID of the seismic event.

    Returns:
        dict: A dictionary containing event information.
    """
    client = Client()
    event = client.get_event(event_id)

    event_info = {
        'time': event.origins[0].time,
        'latitude': event.origins[0].latitude,
        'longitude': event.origins[0].longitude,
        'depth': event.origins[0].depth,
        'magnitude': event.magnitudes[0].mag,
    }

    return event_info


def normalize_trace(data, trace_index=0):
    """
    Normalize a specific trace from the SEG-Y data.

    Args:
        data (np.ndarray): The array containing trace data.
        trace_index (int): The index of the trace to normalize.

    Returns:
        np.ndarray: The normalized trace data.
    """
    if data is not None and trace_index < data.shape[0]:
        trace = data[trace_index, :]
        norm_trace = (trace - np.min(trace)) / (np.max(trace) - np.min(trace))
        return norm_trace
    else:
        print("Invalid trace index or data not available.")
        return None


def calculate_mean_amplitude(data):
    """
    Calculate the mean amplitude of each trace.

    Args:
        data (np.ndarray): The array containing trace data.

    Returns:
        np.ndarray: An array of mean amplitudes for each trace.
    """
    if data is not None:
        return np.mean(data, axis=1)
    else:
        print("Data not available.")
        return None


def calculate_peak_amplitude(data):
    """
    Calculate the peak amplitude of each trace.

    Args:
        data (np.ndarray): The array containing trace data.

    Returns:
        np.ndarray: An array of peak amplitudes for each trace.
    """
    if data is not None:
        return np.max(np.abs(data), axis=1)
    else:
        print("Data not available.")
        return None


def calculate_standard_deviation(data):
    """
    Calculate the standard deviation of each trace.

    Args:
        data (np.ndarray): The array containing trace data.

    Returns:
        np.ndarray: An array of standard deviations for each trace.
    """
    if data is not None:
        return np.std(data, axis=1)
    else:
        print("Data not available.")
        return None


def slice_trace(data, start_sample, end_sample, trace_index=0):
    """
    Slice a specific trace to a specified range of samples.

    Args:
        data (np.ndarray): The array containing trace data.
        start_sample (int): Start sample index.
        end_sample (int): End sample index.
        trace_index (int): The index of the trace to slice.

    Returns:
        np.ndarray: The sliced trace data.
    """
    if data is not None and trace_index < data.shape[0]:
        return data[trace_index, start_sample:end_sample]
    else:
        print("Invalid trace index or data not available.")
        return None


def extract_time_series(data, sample_rate=1000):
    """
    Extract time series from trace data.

    Args:
        data (np.ndarray): The array containing trace data.
        sample_rate (int): Sample rate in Hz.

    Returns:
        np.ndarray: An array of time values corresponding to the samples.
    """
    if data is not None:
        n_samples = data.shape[1]
        return np.arange(n_samples) / sample_rate
    else:
        print("Data not available.")
        return None


def save_to_hdf5(data, output_file):
    """
    Save trace data to an HDF5 file.

    Args:
        data (np.ndarray): The array containing trace data.
        output_file (str): Path to the output HDF5 file.
    """
    try:
        with pd.HDFStore(output_file, mode='w') as store:
            store.put('trace_data', pd.DataFrame(data))
        print(f"Data successfully saved to {output_file}")
    except Exception as e:
        print("An error occurred while saving data:", e)


def load_from_hdf5(input_file):
    """
    Load trace data from an HDF5 file.

    Args:
        input_file (str): Path to the input HDF5 file.

    Returns:
        np.ndarray: An array containing the trace data, or None if an error occurs.
    """
    try:
        with pd.HDFStore(input_file, mode='r') as store:
            data = store['trace_data'].values
        print(f"Data successfully loaded from {input_file}")
        return data
    except Exception as e:
        print("An error occurred while loading data:", e)
        return None


def plot_multiple_traces(data, trace_indices=None):
    """
    Plot multiple traces from the SEG-Y data.

    Args:
        data (np.ndarray): The array containing trace data.
        trace_indices (list of int, optional): List of indices of the traces to plot.
    """
    if data is not None:
        plt.figure(figsize=(10, 6))
        if trace_indices is None:
            trace_indices = range(data.shape[0])  # Plot all traces

        for idx in trace_indices:
            plt.plot(data[idx, :], label=f'Trace {idx + 1}')

        plt.title('Multiple Traces Data')
        plt.xlabel('Sample Number')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()
        plt.show()
    else:
        print("Data not available.")


def create_response(trace, instrument):
    """
    Create a response for the given trace.

    Args:
        trace (Trace): The trace object.
        instrument (str): The instrument type.

    Returns:
        dict: Response object.
    """
    response = trace.stats.response
    response.instrument = instrument
    return response


def remove_response(data, trace):
    """
    Remove the instrument response from the trace data.

    Args:
        data (np.ndarray): The array containing trace data.
        trace (Trace): The trace object.

    Returns:
        np.ndarray: The data with the response removed.
    """
    trace.remove_response(output='displacement')
    return trace.data


def plot_spectrum(data, trace_index=0):
    """
    Plot the spectrum of a specific trace.

    Args:
        data (np.ndarray): The array containing trace data.
        trace_index (int): The index of the trace to plot.
    """
    if data is not None and trace_index < data.shape[0]:
        f, Pxx = welch(data[trace_index, :], fs=1000)
        plt.figure(figsize=(10, 5))
        plt.semilogy(f, Pxx)
        plt.title(f'Spectrum of Trace {trace_index + 1}')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power/Frequency [V**2/Hz]')
        plt.grid()
        plt.show()
    else:
        print("Invalid trace index or data not available.")


def calculate_correlation(data, trace_index1=0, trace_index2=1):
    """
    Calculate the correlation between two traces.

    Args:
        data (np.ndarray): The array containing trace data.
        trace_index1 (int): The index of the first trace.
        trace_index2 (int): The index of the second trace.

    Returns:
        float: The correlation coefficient.
    """
    if data is not None:
        correlation = np.corrcoef(data[trace_index1, :], data[trace_index2, :])[0, 1]
        return correlation
    else:
        print("Data not available.")
        return None


def extract_stream_segment(stream, start_time, end_time):
    """
    Extract a segment of the stream between two times.

    Args:
        stream (Stream): The ObsPy stream.
        start_time (UTCDateTime): Start time of the segment.
        end_time (UTCDateTime): End time of the segment.

    Returns:
        Stream: The extracted segment of the stream.
    """
    return stream.slice(starttime=start_time, endtime=end_time)


def convert_to_segy(stream, output_file):
    """
    Convert an ObsPy Stream to SEG-Y format and save to a file.

    Args:
        stream (Stream): The ObsPy stream to convert.
        output_file (str): Path to the output SEG-Y file.
    """
    # Convert traces to SEG-Y format
    with segyio.create(output_file, segyio.TraceField.DTYPE_FLOAT32) as segy_file:
        for trace in stream:
            # Write trace data and headers
            segy_file.trace[trace.stats.station] = trace.data
            segy_file.header[trace.stats.station] = trace.stats


def get_trace_stats(stream):
    """
    Get statistics for each trace in the stream.

    Args:
        stream (Stream): The ObsPy stream.

    Returns:
        pd.DataFrame: DataFrame containing statistics for each trace.
    """
    stats = []
    for trace in stream:
        stats.append({
            'station': trace.stats.station,
            'mean': np.mean(trace.data),
            'max': np.max(trace.data),
            'min': np.min(trace.data),
            'std': np.std(trace.data),
        })
    return pd.DataFrame(stats)


def filter_data(data, filter_type='bandpass', freq=(0.01, 0.1)):
    """
    Apply a filter to the trace data.

    Args:
        data (np.ndarray): The array containing trace data.
        filter_type (str): The type of filter to apply.
        freq (tuple): Frequency range for bandpass filter.

    Returns:
        np.ndarray: The filtered trace data.
    """
    from obspy.signal.filter import bandpass

    if filter_type == 'bandpass':
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i, :] = bandpass(data[i, :], freq[0], freq[1], df=1000, corners=4, zerophase=True)
        return filtered_data
    else:
        print("Unsupported filter type.")
        return None


def plot_stream(stream):
    """
    Plot all traces in the stream.

    Args:
        stream (Stream): The ObsPy stream to plot.
    """
    plt.figure(figsize=(10, 6))
    for trace in stream:
        plt.plot(trace.times(), trace.data, label=trace.stats.station)
    plt.title('All Traces')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()
