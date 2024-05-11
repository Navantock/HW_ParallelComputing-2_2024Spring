/*
 * Foundations of Parallel Computing II, Spring 2024.
 * Author: Hongfei Wu 
 * Student ID:2301110442.
 * This is a parallel implementation of LU decomposition.
 */

#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>

using namespace std;

namespace utils_serial {
    int N; // number of rows (and columns) in the matrix
    vector<double> mat; // the input matrix
    vector<double> L; // lower triangular matrix
    vector<double> U; // upper triangular matrix

    void abort_with_error_message(const string& msg) {
        cerr << msg << endl;
        abort();
    }

    int read_inputs_and_initialize(const string& filename) {
        ifstream inputf(filename);
        if (!inputf.is_open())
            abort_with_error_message("ERROR: Unable to open input file.");

        inputf >> N;
        mat.resize(N * N);
        for (int i = 0; i < N * N; ++i) {
            inputf >> mat[i];
        }
        inputf.close();

        L.resize(N * N);
        U.resize(N * N);

        copy(mat.begin(), mat.end(), U.begin());

        // Initialize L to identity matrix
        for (int i = 0; i < N; ++i) {
            L[i * N + i] = 1.0;
        }
        return 0;
    }

    void lu_decomposition() {
        for (int j = 0; j < N; ++j) {
            for (int i = j + 1; i < N; ++i) {
                L[i * N + j] = U[i * N + j] / U[j * N + j];
                for (int k = j; k < N; ++k) {
                    U[i * N + k] -= L[i * N + j] * U[j * N + k];
                }
            }
        }
    }

    void write_outputs(const string& output_filename) {
        ofstream outputf(output_filename);
        if (!outputf.is_open())
            abort_with_error_message("ERROR: Unable to open output file.");

        outputf << "L matrix:" << endl;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                outputf << L[i * N + j] << " ";
            }
            outputf << endl;
        }
        outputf << "U matrix:" << endl;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                outputf << U[i * N + j] << " ";
            }
            outputf << endl;
        }
        outputf.close();
    }
}

namespace utils_parallel {
    int N; // number of rows (and columns) in the matrix
    double** mat; // the input matrix
    double** L; // lower triangular matrix
    double** U; // upper triangular matrix


    void abort_with_error_message(const string& msg) {
        cerr << msg << endl;
        abort();
    }

    int read_inputs_and_initialize(const string& filename) {
        ifstream inputf(filename);
        if (!inputf.is_open())
            abort_with_error_message("ERROR: Unable to open input file.");

        inputf >> N;

        // Allocate memory for the matrices
        mat = new double*[N];
        L = new double*[N];
        U = new double*[N];
        for (int i = 0; i < N; ++i) {
            mat[i] = new double[N];
            L[i] = new double[N];
            U[i] = new double[N];
        }

        // Read the matrix from the input file
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                inputf >> mat[i][j];
            }
        }

        inputf.close();

        // Initialize U to the input matrix
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                U[i][j] = mat[i][j];
            }
        }
        // Initialize L to identity matrix
        for (int i = 0; i < N; ++i) {
            L[i][i] = 1.0;
        }

        return 0;
    }

    void lu_decomposition_parallel() {
        for (int j = 0; j < N; ++j) {
            double Ajj = 1 / U[j][j];
            #pragma omp parallel for
            for (int i = j + 1; i < N; ++i){
                L[i][j] = U[i][j] * Ajj;
                for (int k = j; k < N; ++k) {
                    U[i][k] -= L[i][j] * U[j][k];
                }
            }
        }
    }

    void write_outputs(const string& output_filename) {
        ofstream outputf(output_filename);
        if (!outputf.is_open())
            abort_with_error_message("ERROR: Unable to open output file.");

        outputf << "L matrix:" << endl;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                outputf << L[i][j] << " ";
            }
            outputf << endl;
        }
        outputf << "U matrix:" << endl;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                outputf << U[i][j] << " ";
            }
            outputf << endl;
        }
        outputf.close();
    }
}

int main(int argc, char* argv[]) {
    if (argc != 6)
        utils_serial::abort_with_error_message("ERROR: Invalid command. Usage: ./main <input_file> <serial_output_file> <parallel_output_file>");

    string input_filename = argv[1];
    string output_serial_filename = argv[2];
    string output_parallel_filename = argv[3];
    int thread_num = atoi(argv[4]);
    int test_times = atoi(argv[5]);
    double t_serial, t_parallel, t_start, t_end;

    vector<double> serial_time = vector<double>(test_times);
    vector<double> parallel_time = vector<double>(test_times);

    cout << "Test input file: " << input_filename << endl;
    cout << "Test times: " << test_times << endl;
    for (int i = 0; i < test_times; i++) {
        utils_serial::read_inputs_and_initialize(input_filename);
        t_start = omp_get_wtime();
        utils_serial::lu_decomposition();
        t_end = omp_get_wtime();
        t_serial = t_end - t_start;
        serial_time[i] = t_serial;
        utils_serial::write_outputs(output_serial_filename);

        omp_set_num_threads(thread_num);
        utils_parallel::read_inputs_and_initialize(input_filename);
        t_start = omp_get_wtime();
        utils_parallel::lu_decomposition_parallel();
        t_end = omp_get_wtime();
        t_parallel = t_end - t_start;
        parallel_time[i] = t_parallel;
        utils_parallel::write_outputs(output_parallel_filename);
    }
    
    double avg_t_serial = std::accumulate(serial_time.begin(), serial_time.end(), 0.0) / test_times;
    cout << "Average serial time: " << avg_t_serial << "s" << endl;
    double avg_t_parallel = std::accumulate(parallel_time.begin(), parallel_time.end(), 0.0) / test_times;
    cout << "Average parallel time: " << avg_t_parallel << "s" << endl;

    cout << "Speedup: " << avg_t_serial / avg_t_parallel << endl;
    cout << "Efficiency: " << avg_t_serial / avg_t_parallel / thread_num << endl;

    return 0;
}