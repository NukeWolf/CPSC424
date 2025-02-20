// openmp_test.cpp
#include <iostream>
#include <cstdint>
#include <vector>
#include <omp.h>
#include <iomanip>

int main() {
    // Read the integer N (as a 64-bit signed integer) from standard input in binary.
    int64_t N;
    std::cin.read(reinterpret_cast<char*>(&N), sizeof(N));
    if (!std::cin) {
        std::cerr << "Error reading N from input." << std::endl;
        return 1;
    }

    // Allocate arrays:
    // A is a int64 array of length N.
    // B is a int64 array of length 2*N.
    std::vector<int64_t> A(N);
    std::vector<int64_t> B(2 * N);

    // Read array A from binary input.
    std::cin.read(reinterpret_cast<char*>(A.data()), N * sizeof(int64_t));
    if (!std::cin) {
        std::cerr << "Error reading A from input." << std::endl;
        return 1;
    }

    // Read array B from binary input.
    std::cin.read(reinterpret_cast<char*>(B.data()), 2 * N * sizeof(int64_t));
    if (!std::cin) {
        std::cerr << "Error reading B from input." << std::endl;
        return 1;
    }

    // Get the start time.
    double start_time = omp_get_wtime();

    // TODO: your code here. Return the sum in the sum double.
    double sum = 0;

    //Initial
    #pragma omp parallel
    {
        #pragma omp for
        for(int i = 0; i < N; i++){
            A[i] = A[i] + i;
        }

        #pragma omp for
        for(int i = 0; i < N; i++){
            B[i] = B[i] + (i%2);
        }

        #pragma omp for
        for(int i = 0; i < N; i++){
            B[i] = B[i] + A[i] + B[i+N];
        }
        #pragma omp for
        for(int i = N; i < 2*N; i++){
            B[i] = B[i] + 2 * A[i-N] + B[i-N];
        }

        #pragma omp for
        for(int i = 0; i < N; i++){
            A[i] =(B[i] + B[N-i]) / 2.0;
        }

        #pragma omp for reduction(+:sum)
        for(int i = 0; i < N; i++){
            sum += A[i];
        }
    }
    

    

    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;

    // Output the results in the expected format.
    std::cout << std::fixed << std::setprecision(10) <<
        "Final sum: " << sum << std::endl;
    std::cout << "Time taken: " << elapsed << " seconds" << std::endl;

    return 0;
}
