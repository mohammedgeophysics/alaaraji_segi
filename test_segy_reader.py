# tests/test_segy_reader.py

import alaaraji_segi
import unittest
import os


class TestAlaarajiSegy(unittest.TestCase):

    def setUp(self):
        self.valid_file_path = r"C:\Users\My Computer\Videos\h2SW_final.segy"  # تأكد من صحة المسار
        self.invalid_file_path = r"C:\path\to\nonexistent_file.segy"

    def test_read_segy(self):
        data = alaaraji_segi.read_segy(self.valid_file_path, n_samples=100)
        self.assertIsNotNone(data)
        self.assertEqual(data.shape[0], 100)  # تأكد من عدد traces

    def test_inspect_segy(self):
        metadata = alaaraji_segi.inspect_segy(self.valid_file_path)
        self.assertIsNotNone(metadata)
        self.assertIn('number_of_traces', metadata)
        self.assertIn('number_of_samples', metadata)

    def test_invalid_file(self):
        data = alaaraji_segi.read_segy(self.invalid_file_path)
        self.assertIsNone(data)  # يجب أن تعود None عند عدم العثور على الملف

    def test_convert_to_obspy(self):
        data = alaaraji_segi.read_segy(self.valid_file_path, n_samples=100)
        stream = alaaraji_segi.convert_to_obspy(data)
        self.assertIsNotNone(stream)
        self.assertEqual(len(stream), data.shape[0])  # يجب أن يتطابق عدد traces

    def test_export_to_csv(self):
        data = alaaraji_segi.read_segy(self.valid_file_path, n_samples=100)
        output_file = 'output.csv'
        alaaraji_segi.export_to_csv(data, output_file)
        self.assertTrue(os.path.exists(output_file))  # تأكد من أن الملف تم إنشاؤه
        os.remove(output_file)  # حذف الملف بعد الاختبار

    def test_filter_trace(self):
        data = alaaraji_segi.read_segy(self.valid_file_path, n_samples=100)
        filtered_data = alaaraji_segi.filter_trace(data, trace_index=0, threshold=0.01)
        self.assertIsNotNone(filtered_data)  # تأكد من أن البيانات المفلترة ليست None

    def test_normalize_trace(self):
        data = alaaraji_segi.read_segy(self.valid_file_path, n_samples=100)
        normalized_trace = alaaraji_segi.normalize_trace(data, trace_index=0)
        self.assertIsNotNone(normalized_trace)  # تأكد من أن البيانات المعيارية ليست None

    def test_calculate_mean_amplitude(self):
        data = alaaraji_segi.read_segy(self.valid_file_path, n_samples=100)
        mean_amplitudes = alaaraji_segi.calculate_mean_amplitude(data)
        self.assertIsNotNone(mean_amplitudes)  # تأكد من أن النتائج ليست None

    def test_calculate_peak_amplitude(self):
        data = alaaraji_segi.read_segy(self.valid_file_path, n_samples=100)
        peak_amplitudes = alaaraji_segi.calculate_peak_amplitude(data)
        self.assertIsNotNone(peak_amplitudes)  # تأكد من أن النتائج ليست None

    def test_calculate_standard_deviation(self):
        data = alaaraji_segi.read_segy(self.valid_file_path, n_samples=100)
        std_devs = alaaraji_segi.calculate_standard_deviation(data)
        self.assertIsNotNone(std_devs)  # تأكد من أن النتائج ليست None

    def test_slice_trace(self):
        data = alaaraji_segi.read_segy(self.valid_file_path, n_samples=100)
        sliced_trace = alaaraji_segi.slice_trace(data, start_sample=0, end_sample=50, trace_index=0)
        self.assertIsNotNone(sliced_trace)  # تأكد من أن الجزء المقطوع ليس None

    def test_extract_time_series(self):
        data = alaaraji_segi.read_segy(self.valid_file_path, n_samples=100)
        time_series = alaaraji_segi.extract_time_series(data, sample_rate=1000)
        self.assertIsNotNone(time_series)  # تأكد من أن السلسلة الزمنية ليست None

    def test_save_and_load_hdf5(self):
        data = alaaraji_segi.read_segy(self.valid_file_path, n_samples=100)
        output_file = 'output.h5'
        alaaraji_segi.save_to_hdf5(data, output_file)
        loaded_data = alaaraji_segi.load_from_hdf5(output_file)
        self.assertIsNotNone(loaded_data)  # تأكد من أن البيانات المحملة ليست None
        os.remove(output_file)  # حذف الملف بعد الاختبار

    def test_plot_multiple_traces(self):
        data = alaaraji_segi.read_segy(self.valid_file_path, n_samples=100)
        try:
            alaaraji_segi.plot_multiple_traces(data, trace_indices=[0, 1, 2])  # Plot the first three traces
        except Exception as e:
            self.fail(f"plot_multiple_traces raised an exception: {e}")

    def test_get_event_info(self):
        # اختبار الحصول على معلومات حدث زلزالي
        event_info = alaaraji_segi.get_event_info('event_id_here')  # استبدل بـ ID حدث صحيح
        self.assertIsNotNone(event_info)

    def test_create_response(self):
        data = alaaraji_segi.read_segy(self.valid_file_path, n_samples=100)
        stream = alaaraji_segi.convert_to_obspy(data)
        response = alaaraji_segi.create_response(stream[0], 'InstrumentType')  # استبدل بـ instrument الصحيح
        self.assertIsNotNone(response)

    def test_remove_response(self):
        data = alaaraji_segi.read_segy(self.valid_file_path, n_samples=100)
        stream = alaaraji_segi.convert_to_obspy(data)
        corrected_trace = alaaraji_segi.remove_response(data, stream[0])
        self.assertIsNotNone(corrected_trace)

    def test_plot_spectrum(self):
        data = alaaraji_segi.read_segy(self.valid_file_path, n_samples=100)
        try:
            alaaraji_segi.plot_spectrum(data, trace_index=0)  # Plot spectrum of the first trace
        except Exception as e:
            self.fail(f"plot_spectrum raised an exception: {e}")

    def test_calculate_correlation(self):
        data = alaaraji_segi.read_segy(self.valid_file_path, n_samples=100)
        correlation = alaaraji_segi.calculate_correlation(data, trace_index1=0, trace_index2=1)
        self.assertIsNotNone(correlation)  # تأكد من أن معامل الارتباط ليس None

    def test_extract_stream_segment(self):
        data = alaaraji_segi.read_segy(self.valid_file_path, n_samples=100)
        stream = alaaraji_segi.convert_to_obspy(data)
        segment = alaaraji_segi.extract_stream_segment(stream, start_time=UTCDateTime(0), end_time=UTCDateTime(10))
        self.assertIsNotNone(segment)

    def test_convert_to_segy(self):
        data = alaaraji_segi.read_segy(self.valid_file_path, n_samples=100)
        stream = alaaraji_segi.convert_to_obspy(data)
        try:
            alaaraji_segi.convert_to_segy(stream, 'output.segy')
            self.assertTrue(os.path.exists('output.segy'))  # تأكد من أن الملف تم إنشاؤه
            os.remove('output.segy')  # حذف الملف بعد الاختبار
        except Exception as e:
            self.fail(f"convert_to_segy raised an exception: {e}")

    def test_get_trace_stats(self):
        data = alaaraji_segi.read_segy(self.valid_file_path, n_samples=100)
        stream = alaaraji_segi.convert_to_obspy(data)
        stats_df = alaaraji_segi.get_trace_stats(stream)
        self.assertIsNotNone(stats_df)  # تأكد من أن الإحصائيات ليست None

    def test_filter_data(self):
        data = alaaraji_segi.read_segy(self.valid_file_path, n_samples=100)
        filtered_data = alaaraji_segi.filter_data(data, filter_type='bandpass', freq=(0.01, 0.1))
        self.assertIsNotNone(filtered_data)  # تأكد من أن البيانات المفلترة ليست None

    def test_plot_stream(self):
        data = alaaraji_segi.read_segy(self.valid_file_path, n_samples=100)
        stream = alaaraji_segi.convert_to_obspy(data)
        try:
            alaaraji_segi.plot_stream(stream)
        except Exception as e:
            self.fail(f"plot_stream raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()
