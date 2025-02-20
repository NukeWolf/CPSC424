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
    // Initial sum of 0
    double sum = (2 * B[0] + 3 * B[N] + 4 * A[0]) >> 1;
    int max = (N+1)/2;

    // Handle case in the middle to not double middle value
    if (!(N & 1)){
        int i = N / 2;
        B[i] = B[i] + (i & 1) + A[i] + i + B[i+N];
        sum += B[i];
        // Exception to (n mod 2) and (ni mod 2) for even cases of N. Otherwise, for odd cases, this is always 1.
        sum += (max / 2) * 2;
    }
    
    // Handle the constant value
    
    //Initial
    #pragma omp parallel
    {
        
        int n_magic = N & 1;
        int constant = N + n_magic;
        
        #pragma omp for reduction(+:sum) schedule(static)
        for(int i = 1; i < max; i++){
            int ni = N - i;
            sum += ((A[i] + B[i] + B[ni] + A[ni] + B[i+N] + constant + B[ni+N]) >> 1) << 1;
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
