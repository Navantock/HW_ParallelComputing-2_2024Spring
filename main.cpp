#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <vector>
#include <numeric>
#define OUTPUT_TEST_DATA false
#define OUTPUT_RESULT false


namespace utils {
  void mpi_init(int* argc, char*** argv, int* rank, int* size) {
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, size);
  }

  void mpi_finalize() {
    MPI_Finalize();
  }

  void prefix_sum(long* a, long* prefix_sum, long N) {
    prefix_sum[0] = a[0];
    for (long i = 1; i < N; i++) {
      prefix_sum[i] = prefix_sum[i - 1] + a[i];
    }
  }
}

int main(int argc, char* argv[]) {
  long N;
  int rank, size, test_times;

  if (argc < 3) {
    test_times = 10;
  }
  else {
    test_times = std::atoi(argv[2]);
  }
  if (argc < 2) {
    N = 32;
  }
  else {
    N = std::atol(argv[1]);
  }

  // MPI initialization
  utils::mpi_init(&argc, &argv, &rank, &size);

  if (rank == 0) {
    std::cout << "---PREFIX SUM EXPERIMENT---" << std::endl;
    std::cout << "Array Size N = " << N << std::endl;
    std::cout << "Thread Number = " << size << std::endl;
    std::cout << "Test Times = " << test_times << std::endl;
    std::cout << "---------------------------" << std::endl;
  }

  // Parameters
  double start_time, end_time;
  long* a;


  /* Serial Version */
  long* serial_results;
  std::vector<double> serial_time;
  if (rank == 0) {
    // Randomly initialize data
    std::srand(0);
    a = (long*)malloc(sizeof(long) * N);
    for (long i = 0; i < N; i++) {
      a[i] = std::rand() % 100;
    }
    #if OUTPUT_TEST_DATA
    // Print test data
    std::cout << "Test data:\n";
    for (long i = 0; i < N; i++) {
      std::cout << a[i] << " ";
    }
    std::cout << "\n";
    #endif

    std::cout << "Serial Version\n";
    // Calculation
    serial_results = (long*)malloc(sizeof(long) * N);
    for (int i = 0; i < test_times; i++) {
      double start_time = MPI_Wtime();
      utils::prefix_sum(a, serial_results, N);
      double end_time = MPI_Wtime();
      serial_time.push_back(end_time - start_time);
    }
    std::cout << "Mean Time: " << std::accumulate(serial_time.begin(), serial_time.end(), 0.0) / test_times << "s\n";
    #if OUTPUT_RESULT
    // Print Result
    for (long i = 0; i < N; i++) {
      std::cout << serial_results[i] << " ";
    }
    #endif
    std::cout << "Serial End" << std::endl << "---------------------------" << std::endl;
  }


  /* Parallel Version */
  if (rank == 0) {
    std::cout << "Parallel Version\n";
  }
  // Task division
  long* parallel_results;
  std::vector<double> parallel_time;
  if (rank == 0) {
    parallel_results = (long*)malloc(sizeof(long) * N);
  }
  long mod_num = N % size;
  long start = N / size * rank + (rank < mod_num ? rank : mod_num);
  long end = start + N / size + (rank < mod_num ? 1 : 0);
  long local_length = end - start;
  long* prefix_sum = (long*)malloc(sizeof(long) * local_length);
  long* seg_last_num_addr = prefix_sum + local_length - 1;
  long add_num;

  // Scatter data
  long* a_seg = (long*)malloc(sizeof(long) * local_length);
  int* send_counts = (int*)malloc(sizeof(int) * size);
  int* displs = (int*)malloc(sizeof(int) * size);
  MPI_Allgather(&local_length, 1, MPI_INT, send_counts, 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(&start, 1, MPI_INT, displs, 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Scatterv(a, send_counts, displs, MPI_LONG, a_seg, local_length, MPI_LONG, 0, MPI_COMM_WORLD);

  // Calculation
  for (int i = 0; i < test_times; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      start_time = MPI_Wtime();
    }
    utils::prefix_sum(a_seg, prefix_sum, local_length);
    MPI_Scan(seg_last_num_addr, &add_num, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    add_num -= *seg_last_num_addr;
    for (long i = 0; i < local_length; i++) {
      prefix_sum[i] += add_num;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      end_time = MPI_Wtime();
      parallel_time.push_back(end_time - start_time);
    }
  }
  // Gather result
  MPI_Gatherv(prefix_sum, local_length, MPI_LONG, parallel_results, send_counts, displs, MPI_LONG, 0, MPI_COMM_WORLD);
  // Print result
  if (rank == 0) {
    std::cout << "Mean Time: " << std::accumulate(parallel_time.begin(), parallel_time.end(), 0.0) / test_times << "s\n";
    std::cout << "Parallel End" << std::endl << "---------------------------" << std::endl;
    // Check correctness
    std::cout << "Checking correctness...\n";
    bool correct = true;
    for (long i = 0; i < N; i++) {
      if (parallel_results[i] != serial_results[i]) {
        correct = false;
        break;
      }
    }
    if (correct) {
      std::cout << "Correct!\n";
    }
    else {
      std::cout << "Incorrect!\n";
    }
    std::cout << "---------------------------" << std::endl;

    free(a);
    free(serial_results);
    free(parallel_results);
  }
  free(a_seg);
  free(prefix_sum);
  free(send_counts);
  free(displs);
  
  if (rank == 0) {
    std::cout << "Prallel Performance:" << std::endl;
    std::cout << "Speedup: " << std::accumulate(serial_time.begin(), serial_time.end(), 0.0) / std::accumulate(parallel_time.begin(), parallel_time.end(), 0.0) << std::endl;
    std::cout << "Efficiency: " << std::accumulate(serial_time.begin(), serial_time.end(), 0.0) / std::accumulate(parallel_time.begin(), parallel_time.end(), 0.0) / size << std::endl;
    std::cout << "------------END------------" << std::endl;
  }
  utils::mpi_finalize();
  return 0;
}