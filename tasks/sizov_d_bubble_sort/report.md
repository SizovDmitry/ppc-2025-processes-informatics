# Сортировка пузырьком (алгоритм чет-нечетной перестановки) / Bubble Sort (Odd–Even Transposition Algorithm)

- Student: Сизов Дмитрий Игоревич, group 3823Б1ФИ2
- Technology: SEQ + MPI
- Variant: 21

## 1. Introduction

The objective of this work is to implement both sequential and parallel (MPI-based) versions of bubble sort using the Odd–Even Transposition Sort algorithm and to compare their correctness and performance. The task belongs to the class of comparison-based sorting algorithms that rely on pairwise exchanges of adjacent elements. The expected outcome is a fully correct implementation of the sorting procedure, successful execution of all functional and performance tests in the PPC framework, and a demonstrated speedup of the MPI version on large input data.

## 2. Problem Statement

Given: an array of integers (`InType = std::vector<int>`).

Required:

- Sort the array in non-decreasing order.
- Provide two implementations: a sequential version and an MPI-based version.
- Both implementations must operate correctly on all provided tests.

Input/output format:

- Input: `std::vector<int>`
- Output: a sorted `std::vector<int>`

Constraint: empty input is not allowed.

## 3. Baseline Algorithm (Sequential)

The sequential version uses the classical odd–even sort algorithm.  
Odd–even sort operates as a sequence of phases, where each phase compares adjacent elements, but the pairs differ between even and odd phases.

During an **even phase**, the algorithm processes element pairs starting at even indices: (0,1), (2,3), (4,5), and so on.  
For each pair, if the left element is greater than the right one, the two elements are swapped.

During an **odd phase**, the algorithm processes pairs starting at odd indices: (1,2), (3,4), (5,6), etc.  
The same swapping rule applies.

These phases alternate, and due to the regular shift of comparison windows, any elements placed in the wrong order gradually move toward their correct positions.  
The algorithm continues alternating phases until a pass occurs with no swaps, which indicates that the array is fully sorted.

Time complexity: **O(n²)**.

## 4. Parallelization Scheme (MPI)

### 4.1 Data Distribution

At the beginning of the parallel algorithm, the root process (rank 0) broadcasts  
the total number of elements `n` using `MPI_Bcast`, so that every process knows  
the global problem size before receiving its portion of data.

To split the input array among all MPI processes, the helper function  
`ComputeScatterInfo` constructs two arrays:

• `counts[i]` — how many elements are assigned to process *i*  
• `displs[i]` — the starting index of process *i* in the global array

The partitioning is almost uniform: each process receives either  
`⌊n / p⌋` or `⌊n / p⌋ + 1` elements, with the first `n % p` processes obtaining  
one additional element to compensate for uneven division.

These arrays are then used by `MPI_Scatterv`, which distributes the corresponding  
contiguous segments of the input array to all processes.  
As a result, each process obtains its own local block and can participate  
in the odd–even transposition phases.

### 4.2 Local Work

After data distribution, each process receives its segment of the global array  
into the vector `local`. Since we are implementing the classical odd–even  
transposition sort (a distributed analogue of bubble sort), no initial local  
sorting is performed: each process works with its raw unsorted block.

During every global phase, each process performs two operations:

• a **local odd–even pass**, comparing adjacent elements inside its block  
  according to the global parity of the phase (even or odd indices),  

• a **boundary exchange** with exactly one neighbor (left or right),  
  where the processes exchange a single boundary element and keep the  
  appropriate one (left keeps the smaller, right keeps the larger).

Together, these steps replicate the behavior of bubble sort across process  
boundaries: small values gradually move left, large values move right.

No merging of whole blocks and no local full sorts are performed.  
Global order emerges progressively as the algorithm executes `n` odd–even phases.

### 4.3 Neighbor Exchanges (Odd–Even Transposition Phases)

The MPI version implements the classical **odd–even transposition sort**, which is
a parallel analogue of bubble sort. The algorithm proceeds through **`n` global
phases**, where `n` is the total number of elements. This number of phases is
sufficient to guarantee that every element can “bubble” to its correct global
position.

Each global phase consists of two coordinated actions performed by every process:

1. **Determine phase parity**

   The phase number (even or odd) defines which global index pairs must be
   compared during this pass. Processes use this parity to decide which local
   adjacent positions should be compared and potentially swapped.

2. **Local comparisons**

   Each process performs a local odd–even pass on its own block:
   it compares adjacent elements whose global indices match the phase parity and
   swaps them if necessary. This is the exact analogue of the local step of
   bubble sort.

3. **Boundary exchange with neighbor**

   Because global indices run across process boundaries, each phase also
   requires comparing elements on the border between two neighboring blocks.
   A single neighbor is chosen deterministically:

   • in even phases, even-ranked processes communicate with rank+1  
   • in odd phases, even-ranked processes communicate with rank−1  
   (odd ranks perform the symmetric counterpart)

   Processes exchange one boundary element using `MPI_Sendrecv`.  
   Then:

   • the left process keeps the *minimum* of the two values,  
   • the right process keeps the *maximum*.

   This reproduces the movement of small elements leftward and large elements
   rightward, exactly as in bubble sort.

If a process has no valid neighbor in the current phase (partner out of range or
empty block), it simply skips the boundary exchange.

Repeating these steps for all `n` phases ensures that every element has enough
opportunities to move across all processes. As a result, the full distributed
array becomes globally sorted without any additional merging steps.

### 4.4 Gathering and Finalization

After all odd–even transposition phases have finished, every process holds a block
that is internally sorted and already consistent with its neighbors. No further
comparisons or adjustments are required.

The finalization stage is straightforward and handled by the root process:

1. **Collecting results**

   Each process sends its local block to rank 0 using `MPI_Gatherv`.  
   The root reconstructs the global array using the same `counts` and `displs`
   computed during data distribution, ensuring that all blocks return to their
   correct positions in the overall sequence.

2. **Producing the final output**

   The concatenated array on the root process is already fully sorted, because
   odd–even transposition guarantees global correctness after performing all
   required phases. No extra merging or post‐processing is needed.

At this point, rank 0 holds the complete sorted array, which becomes the final
output of the MPI sorting procedure.

## 5. Implementation Details

### 5.1 Project Structure

```cpp
tasks/sizov_d_bubble_sort
├── common
│   └── include
│       └── common.hpp
├── data
│   ├── test1.txt
│   ├── test2.txt
│   ├── test3.txt
│   ├── ...
│   └── test40.txt
├── mpi
│   ├── include
│   │   └── ops_mpi.hpp
│   └── src
│       └── ops_mpi.cpp
├── seq
│   ├── include
│   │   └── ops_seq.hpp
│   └── src
│       └── ops_seq.cpp
├── tests
│   ├── functional
│   │   └── main.cpp
│   └── performance
│       └── main.cpp
├── info.json
├── report.md
└── settings.json

```

### 5.2 Data

The functional tests use a set of input files located in:
  `tasks/sizov_d_bubble_sort/data/testN.txt`

Each file contains two lines:  

1. an unsorted array,  
2. the reference sorted result.

### 5.3 Additional Helper Functions

Several auxiliary helper functions support the MPI implementation.  
All algorithm-specific helpers are located inside an anonymous namespace in  
`mpi/src/ops_mpi.cpp`, while `TrimString` belongs to the functional test utilities.

- **`ComputeScatterInfo(total, size, counts, displs)`**  
  Computes how many elements each process receives (`counts[i]`) and where its block  
  begins in the global array (`displs[i]`).  
  Handles uneven division by distributing one extra element to the first  
  `total % size` processes.

- **`LocalOddEvenPass(local, global_start, parity)`**  
  Performs one local bubble-style pass inside a process’s block.  
  Only pairs whose *global* indices match the current phase parity are compared  
  and swapped.  
  This is the local analogue of the odd–even bubble-sort step.

- **`ExchangeBoundary(local, counts, rank, partner)`**  
  Handles the communication between neighboring processes.  
  Only one boundary value is exchanged:  
  the rightmost element from the left block and the leftmost from the right block.  
  After `MPI_Sendrecv`, each side updates its boundary element according to  
  bubble-sort semantics (left keeps the minimum; right keeps the maximum).

- **`OddEvenPhase(local, counts, displs, rank, size, phase)`**  
  Performs one full odd–even phase consisting of:  
  1) local comparisons via `LocalOddEvenPass`, and  
  2) a single boundary exchange with the appropriate neighbor determined  
     by phase parity and process rank.

- **`TrimString(std::string& s)`** *(functional test utility)*  
  Removes trailing and leading whitespace characters (`\r`, `\n`, `\t`, spaces)  
  before test input parsing to avoid formatting-related errors.

Each helper function isolates a well-defined step of the algorithm, improving  
the clarity and maintainability of the overall MPI bubble-sort implementation.

## 6. Experimental Setup

| Component  | Value                                   |
|------------|-----------------------------------------|
| CPU        | 12th Gen Intel(R) Core(TM) i5-12450H    |
| RAM        | 16 GB                                   |
| ОС         | OS: Ubuntu 24.04 (DevContainer / WSL2)  |
| Compiler   | GCC 13.3.0 (g++), C++20, CMake, Release |
| MPI        | mpirun (Open MPI) 4.1.6                 |

## 7. Results and Discussion

### 7.1 Correctness

Correctness has been verified through:

- multiple functional tests,
- comparison against `std::ranges::sort`,
- matching results between the MPI and SEQ implementations.

### 7.2 Performance

For local performance measurements and for comparing the sequential and parallel versions, a unified input size of 75,000/175,000/275,000 elements was used.  
The performance tests were executed locally on my machine with the hardware configuration listed above.  

The speedup is computed as:

$$S(p) = \frac{T_{seq}}{T_{p}}$$

The efficiency is computed as:

$$E(p) = \frac{S(p)}{p} \cdot 100\%$$

**Performance (pipeline, size = 75,000):**

| Mode | Processes | Time (s)       | Speedup S(p) | Efficiency E(p) |
|------|-----------|----------------|--------------|-----------------|
| SEQ  | 1         | 1.1382211208   | 1.0000       | N/A             |
| MPI  | 2         | 1.4889316424   | 0.7645       | 38.22%          |
| MPI  | 4         | 1.0739249196   | 1.0599       | 26.50%          |
| MPI  | 8         | 1.8178929094   | 0.6261       | 7.83%           |

---

**Performance (task_run, size = 75,000):**

| Mode | Processes | Time (s)       | Speedup S(p) | Efficiency E(p) |
|------|-----------|----------------|--------------|-----------------|
| SEQ  | 1         | 0.2347642899   | 1.0000       | N/A             |
| MPI  | 2         | 1.5365564150   | 0.1528       | 7.64%           |
| MPI  | 4         | 1.1454324088   | 0.2050       | 5.12%           |
| MPI  | 8         | 1.8488088746   | 0.1270       | 1.59%           |

---

**Performance (pipeline, size = 175,000):**

| Mode | Processes | Time (s)       | Speedup S(p) | Efficiency E(p) |
|------|-----------|----------------|--------------|-----------------|
| SEQ  | 1         | 6.4017530918   | 1.0000       | N/A             |
| MPI  | 2         | 7.9087940308   | 0.8090       | 40.45%          |
| MPI  | 4         | 5.8502519922   | 1.0945       | 27.36%          |
| MPI  | 8         | 9.7293114716   | 0.6578       | 8.22%           |

---

**Performance (task_run, size = 175,000):**

| Mode | Processes | Time (s)       | Speedup S(p) | Efficiency E(p) |
|------|-----------|----------------|--------------|-----------------|
| SEQ  | 1         | 1.4572491646   | 1.0000       | N/A             |
| MPI  | 2         | 8.2717706478   | 0.1762       | 8.81%           |
| MPI  | 4         | 5.9481297592   | 0.2449       | 6.12%           |
| MPI  | 8         | 9.3558528120   | 0.1558       | 1.95%           |

---

**Performance (pipeline, size = 275,000):**

| Mode | Processes | Time (s)       | Speedup S(p) | Efficiency E(p) |
|------|-----------|----------------|--------------|-----------------|
| SEQ  | 1         | 14.0848023891  | 1.0000       | N/A             |
| MPI  | 2         | 19.9201253816  | 0.7071       | 35.35%          |
| MPI  | 4         | 12.4546995072  | 1.1309       | 28.27%          |
| MPI  | 8         | 14.0781657308  | 1.0005       | 12.51%          |

---

**Performance (task_run, size = 275,000):**

| Mode | Processes | Time (s)       | Speedup S(p) | Efficiency E(p) |
|------|-----------|----------------|--------------|-----------------|
| SEQ  | 1         | 2.8867703438   | 1.0000       | N/A             |
| MPI  | 2         | 19.5067180478  | 0.1480       | 7.40%           |
| MPI  | 4         | 11.8059592702  | 0.2445       | 6.11%           |
| MPI  | 8         | 12.9787113786  | 0.2224       | 2.78%           |

## 8. Conclusions

Both the sequential (SEQ) and MPI versions of the odd–even bubble sort were fully implemented.

## 9. References

- Гергель В.П. Высокопроизводительные вычисления для многопроцессорных многоядерных систем. Серия «Суперкомпьютерное образование».
- Гергель В.П. Теория и практика параллельных вычислений. – М.: Интуит Бином. Лаборатория знаний, 2007.
- Wilkinson B., Allen M. Parallel programming: techniques and applications using networked workstations and parallel computers. – Prentice Hall, 1999.

## Appendix

```cpp
bool SizovDBubbleSortMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = (rank == 0 ? static_cast<int>(data_.size()) : 0);
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n <= 1) {
    if (rank != 0) {
      data_.assign(n, 0);
    }
    if (n > 0) {
      MPI_Bcast(data_.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
    }
    GetOutput() = data_;
    return true;
  }

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  ComputeScatterInfo(n, size, counts, displs);

  const int local_n = counts[rank];
  std::vector<int> local(local_n);

  MPI_Scatterv((rank == 0 ? data_.data() : nullptr),
               counts.data(), displs.data(),
               MPI_INT,
               local.data(), local_n,
               MPI_INT,
               0, MPI_COMM_WORLD);

  for (int phase = 0; phase < n; ++phase) {
    OddEvenPhase(local, counts, displs, rank, size, phase);
  }

  std::vector<int> result(n);
  MPI_Gatherv(local.data(), local_n, MPI_INT,
              result.data(),
              counts.data(), displs.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = result;
  } else {
    GetOutput().assign(n, 0);
  }

  if (n > 0) {
    MPI_Bcast(GetOutput().data(), n, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return true;
}
```
