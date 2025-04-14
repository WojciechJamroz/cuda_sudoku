import cupy as cp
import numpy as np # Still useful for initial setup and comparison
import time
import math
import datetime # Needed for formatting timedelta
import multiprocessing # For CPU core count
import os # For environment variables
import argparse # For command line arguments

# --- Set CUDA optimization environment variables ---
os.environ['CUDA_LAUNCH_BLOCKING'] = '0' # Async execution
os.environ['CUPY_GPU_MEMORY_LIMIT'] = '0' # No memory limit

# --- CuPy Kernel for Validity Checking ---
# Enhanced version with improved early termination and memory access patterns

cuda_check_kernel_code = r'''
#define ALL_SEEN_MASK 0x3FE // Binary 1111111110 (bits 1 through 9 set)

extern "C" __global__ void check_sudoku_batch(
    const signed char* __restrict__ boards, // Input: Flattened batch of boards (B, 9, 9)
    bool* __restrict__ results,             // Output: Boolean result for each board (B,)
    int batch_size)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x; // Board index

    if (b >= batch_size) {
        return; // Out of bounds check
    }

    // Pointer to the start of the current board in the flattened array
    const signed char* __restrict__ current_board = boards + b * 81;
    
    // ---- Optimized with shared memory for better cache locality ----
    __shared__ signed char shared_board[81 * 32]; // Supports up to 32 boards per block
    
    // Thread index within the block
    int thread_idx = threadIdx.x;
    int block_size = blockDim.x;
    
    // Load the board into shared memory
    int local_board_offset = (thread_idx * 81) % (81 * 32); // Wrap around at 32 boards
    
    // Collaborative loading to shared memory
    for (int i = 0; i < 81; i += block_size) {
        int idx = thread_idx + i;
        if (idx < 81) {
            shared_board[local_board_offset + idx] = current_board[idx];
        }
    }
    __syncthreads(); // Ensure all threads have loaded shared memory
    
    // Get pointer to this board's data in shared memory
    const signed char* __restrict__ board = shared_board + local_board_offset;
    
    // Early validity flag - terminate as soon as we find a violation
    bool is_valid = true;

    // --- Check Rows with early termination ---
    for (int r = 0; r < 9 && is_valid; ++r) {
        unsigned short seen_mask = 0; // Reset mask for the row
        
        #pragma unroll
        for (int c = 0; c < 9 && is_valid; ++c) {
            signed char val = board[r * 9 + c];
            
            // Value must be 1-9
            is_valid = is_valid && (val >= 1 && val <= 9);
            if (!is_valid) break;

            unsigned short val_bit = 1 << val; // Bit corresponding to the value
            
            // Check for duplicates
            bool has_duplicate = (seen_mask & val_bit) != 0;
            is_valid = is_valid && !has_duplicate;
            if (!is_valid) break;
            
            seen_mask |= val_bit; // Set the bit for this value
        }
        
        // Optional full check: all digits 1-9 must be present
        // is_valid = is_valid && (seen_mask == ALL_SEEN_MASK);
        // Commented out because we're just checking validity, not completeness
    }

    // --- Check Columns (only if rows are valid) ---
    for (int c = 0; c < 9 && is_valid; ++c) {
        unsigned short seen_mask = 0; // Reset mask for the column
        
        #pragma unroll
        for (int r = 0; r < 9 && is_valid; ++r) {
            signed char val = board[r * 9 + c];
            unsigned short val_bit = 1 << val;
            
            // Check for duplicates (value range already verified in row check)
            bool has_duplicate = (seen_mask & val_bit) != 0;
            is_valid = is_valid && !has_duplicate;
            if (!is_valid) break;
            
            seen_mask |= val_bit;
        }
        
        // Optional full check
        // is_valid = is_valid && (seen_mask == ALL_SEEN_MASK);
    }

    // --- Check 3x3 Blocks (only if rows and columns are valid) ---
    for (int box_r = 0; box_r < 9 && is_valid; box_r += 3) {
        for (int box_c = 0; box_c < 9 && is_valid; box_c += 3) {
            unsigned short seen_mask = 0; // Reset mask for the block
            
            #pragma unroll 9
            for (int i = 0; i < 9 && is_valid; ++i) {
                int r = box_r + (i / 3);
                int c = box_c + (i % 3);
                signed char val = board[r * 9 + c];
                unsigned short val_bit = 1 << val;
                
                // Check for duplicates
                bool has_duplicate = (seen_mask & val_bit) != 0;
                is_valid = is_valid && !has_duplicate;
                if (!is_valid) break;
                
                seen_mask |= val_bit;
            }
            
            // Optional full check
            // is_valid = is_valid && (seen_mask == ALL_SEEN_MASK);
        }
    }

    results[b] = is_valid;
}
'''

# --- Separate kernel for smart random filling ---
cuda_smart_fill_kernel_code = r'''
// --- Smart Random Generation Kernel ---
// Optimized kernel for constraint-based smart filling
extern "C" __global__ void smart_random_fill(
    signed char* __restrict__ boards,          // Output: Board batch to fill
    const int* __restrict__ empty_row_indices, // Input: Row indices of empty cells
    const int* __restrict__ empty_col_indices, // Input: Column indices of empty cells
    const int* __restrict__ board_indices,     // Input: Board indices for each empty cell
    const int num_empty_cells,                 // Input: Number of empty cells per board
    const int batch_size,                      // Input: Number of boards in batch
    const unsigned int seed)                   // Input: Random seed
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_empty_cells) return;
    
    // Determine which board and which empty cell this thread handles
    int board_idx = board_indices[idx];
    int row = empty_row_indices[idx % num_empty_cells];
    int col = empty_col_indices[idx % num_empty_cells];
    
    // Use shared memory to cache valid options
    __shared__ unsigned short constraints[1024]; // Assuming max 1024 threads per block
    __shared__ signed char options[1024 * 9];   // Max 9 options per thread
    
    // Pointer to current board
    signed char* current_board = boards + board_idx * 81;
    
    // Calculate thread's position in block for shared memory access
    int thread_pos = threadIdx.x;
    
    // ---- Highly optimized constraint calculation ----
    
    // Use bit manipulation for faster constraints checking
    unsigned short invalid_mask = 0;
    
    // ---- Row constraints with vectorized loading ----
    #pragma unroll
    for (int c = 0; c < 9; c++) {
        signed char val = current_board[row * 9 + c];
        if (val >= 1 && val <= 9) {
            invalid_mask |= (1 << val);
        }
    }
    
    // ---- Column constraints with vectorized loading ----
    #pragma unroll
    for (int r = 0; r < 9; r++) {
        signed char val = current_board[r * 9 + col];
        if (val >= 1 && val <= 9) {
            invalid_mask |= (1 << val);
        }
    }
    
    // ---- Box constraints with vectorized loading ----
    int box_r = (row / 3) * 3;
    int box_c = (col / 3) * 3;
    
    #pragma unroll
    for (int r = box_r; r < box_r + 3; r++) {
        #pragma unroll
        for (int c = box_c; c < box_c + 3; c++) {
            signed char val = current_board[r * 9 + c];
            if (val >= 1 && val <= 9) {
                invalid_mask |= (1 << val);
            }
        }
    }
    
    // Save constraints to shared memory for later counting
    constraints[thread_pos] = invalid_mask;
    
    // Count valid options (values 1-9 not in the invalid_mask)
    int valid_count = 0;
    #pragma unroll
    for (int val = 1; val <= 9; val++) {
        if (!(invalid_mask & (1 << val))) {
            options[thread_pos * 9 + valid_count] = val;
            valid_count++;
        }
    }
    
    // Generate random value with advanced hashing for better distribution
    unsigned int hash = seed + idx;
    hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
    hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
    hash = (hash >> 16) ^ hash;
    
    // If we have valid options, pick one; otherwise use completely random value
    if (valid_count > 0) {
        // Choose from valid options with better randomness
        int choice = hash % valid_count;
        current_board[row * 9 + col] = options[thread_pos * 9 + choice];
    } else {
        // No valid options, pick a completely random value as fallback
        current_board[row * 9 + col] = (hash % 9) + 1;
    }
}
'''

# --- Additional optimized kernel for advanced solving ---
cuda_advanced_kernel_code = r'''
extern "C" __global__ void advanced_smart_fill(
    signed char* __restrict__ boards,          // Output: Board batch to fill
    const int* __restrict__ empty_row_indices, // Input: Row indices of empty cells
    const int* __restrict__ empty_col_indices, // Input: Column indices of empty cells 
    const int* __restrict__ board_indices,     // Input: Board indices for each empty cell
    const int num_empty_cells,                 // Input: Number of empty cells per board
    const int batch_size,                      // Input: Number of boards in batch
    const unsigned int seed,                   // Input: Random seed
    const float cell_difficulty_bias)          // Input: Bias factor for difficult cells (1.0-10.0)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_empty_cells) return;
    
    // Determine which board and which empty cell this thread handles
    int board_idx = board_indices[idx];
    int cell_idx = idx % num_empty_cells;
    int row = empty_row_indices[cell_idx];
    int col = empty_col_indices[cell_idx];
    
    // Pointer to current board
    signed char* current_board = boards + board_idx * 81;
    
    // --- Advanced constraint calculation ---
    // Use bit manipulation for faster constraints checking
    unsigned short invalid_mask = 0;
    
    // Row constraints
    #pragma unroll
    for (int c = 0; c < 9; c++) {
        signed char val = current_board[row * 9 + c];
        if (val >= 1 && val <= 9) {
            invalid_mask |= (1 << val);
        }
    }
    
    // Column constraints
    #pragma unroll
    for (int r = 0; r < 9; r++) {
        signed char val = current_board[r * 9 + col];
        if (val >= 1 && val <= 9) {
            invalid_mask |= (1 << val);
        }
    }
    
    // Box constraints
    int box_r = (row / 3) * 3;
    int box_c = (col / 3) * 3;
    
    #pragma unroll
    for (int r = box_r; r < box_r + 3; r++) {
        #pragma unroll
        for (int c = box_c; c < box_c + 3; c++) {
            signed char val = current_board[r * 9 + c];
            if (val >= 1 && val <= 9) {
                invalid_mask |= (1 << val);
            }
        }
    }
    
    // Count valid options (values 1-9 not in invalid_mask)
    int valid_count = 0;
    signed char valid_options[9]; // Store the valid options
    
    #pragma unroll
    for (int val = 1; val <= 9; val++) {
        if (!(invalid_mask & (1 << val))) {
            valid_options[valid_count++] = val;
        }
    }
    
    // --- Advanced filling strategy ---
    // Generate multiple random seeds for better distribution
    unsigned int hash = seed + idx;
    hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
    hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
    hash = (hash >> 16) ^ hash;
    
    // Second hash for additional randomness in difficult cases
    unsigned int hash2 = seed + idx * 1664525 + 1013904223;
    hash2 = ((hash2 >> 16) ^ hash2) * 0x3B9A;
    hash2 = (hash2 >> 16) ^ hash2;
    
    if (valid_count > 0) {
        // Normal case: choose from valid options
        int choice_idx = hash % valid_count;
        
        // Use difficulty bias to prefer certain values for difficult cells
        // This helps with avoiding local maxima in the search space
        if (valid_count > 1 && cell_difficulty_bias > 1.0f) {
            // Get a second choice and probabilistically pick between them
            int alt_choice_idx = hash2 % valid_count;
            if (alt_choice_idx != choice_idx) {
                // Use cell difficulty as bias - higher bias means more randomness
                float rand_val = (float)(hash ^ hash2) / (float)(0xFFFFFFFF);
                if (rand_val < (1.0f / cell_difficulty_bias)) {
                    choice_idx = alt_choice_idx; // Use alternative choice
                }
            }
        }
        
        current_board[row * 9 + col] = valid_options[choice_idx];
    } else {
        // No valid options, use completely random value
        current_board[row * 9 + col] = (hash % 9) + 1;
    }
}
'''

# Compile the kernels - now with separate source code for each kernel
check_sudoku_batch_kernel = cp.RawKernel(cuda_check_kernel_code, 'check_sudoku_batch')
smart_random_fill_kernel = cp.RawKernel(cuda_smart_fill_kernel_code, 'smart_random_fill')
advanced_smart_fill_kernel = cp.RawKernel(cuda_advanced_kernel_code, 'advanced_smart_fill')

# --- Helper Functions ---

def print_board(board):
    """Prints the Sudoku board (works with NumPy or CuPy arrays after .get())."""
    if isinstance(board, cp.ndarray):
        board = board.get() # Get NumPy array from CuPy array
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - ")
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")
            print(int(board[i, j]) if board[i, j] != 0 else ".", end=" ") # Use int() for clean printing
        print()

# --- Main GPU Accelerated Solving Function ---

def solve_sudoku_randomly_gpu(puzzle_board_list, batch_size=1_048_576, max_attempts=None):
    """
    Attempts to solve Sudoku using random generation and checking accelerated on GPU.

    Args:
        puzzle_board_list: The initial puzzle as a 9x9 Python list of lists.
        batch_size: How many random boards to generate and check in parallel per iteration.
                    Adjust based on GPU memory (VRAM). Powers of 2 often good.
        max_attempts: Stop after this many total attempts (optional).
    """
    print(f"--- GPU Accelerated Random Sudoku Solver ---")
    print(f"Batch size: {batch_size:,}")

    # --- Initialization ---
    start_time_setup = time.time()
    puzzle_board_np = np.array(puzzle_board_list, dtype=np.int8)
    print("Initial Puzzle:")
    print_board(puzzle_board_np) # Use NumPy version for printing setup

    # Find empty cells using NumPy initially
    empty_cells_np = np.argwhere(puzzle_board_np == 0) # Gets Nx2 array of [row, col]
    num_empty_cells = len(empty_cells_np)
    num_combinations = 0 # Initialize

    if num_empty_cells == 0:
        print("Board is already full.")
        # Perform a single check (can be done on CPU or GPU)
        temp_gpu_board = cp.array(puzzle_board_np).reshape(1, 9, 9)
        temp_results = cp.zeros(1, dtype=bool)
        check_sudoku_batch_kernel((1,), (1,), (temp_gpu_board, temp_results, 1))
        if temp_results.get()[0]:
             print("And it's a valid solution!")
             return puzzle_board_np, 0
        else:
             print("But it's not a valid solution.")
             return None, 0

    print(f"\nFound {num_empty_cells} empty cells to fill.")

    # Calculate and print the theoretical number of combinations
    if num_empty_cells > 0:
        # Use float for potentially huge numbers, though Python handles large ints
        num_combinations = float(9**num_empty_cells)
        print(f"Theoretical number of combinations to fill empty cells: 9^{num_empty_cells} = {num_combinations:,.0f}")
        # Add a warning if the number is astronomically large
        if num_combinations > 1e18: # Arbitrary threshold for 'very large'
             print("Note: This is a vast number of combinations. Finding a solution relies on probability, not exhaustive search.")
    else:
        # This case is handled above, but for completeness:
        print("Theoretical number of combinations: 1 (board is already full)")
        num_combinations = 1.0

    # --- Prepare GPU Data ---
    # Send initial board state and empty cell coordinates to GPU
    puzzle_gpu = cp.array(puzzle_board_np, dtype=cp.int8)
    empty_cells_gpu = cp.array(empty_cells_np, dtype=cp.int32) # Use int32 for indices

    # Pre-calculate indices for filling batches efficiently
    # We need to tell CuPy *which* board and *which* cell to put each random number into.
    # Create board indices [0, 0, ..., 0, 1, 1, ..., 1, ..., B-1, B-1, ..., B-1]
    board_indices = cp.repeat(cp.arange(batch_size, dtype=cp.int32), num_empty_cells)
    # Tile the empty cell row/col indices B times
    empty_rows = cp.tile(empty_cells_gpu[:, 0], batch_size)
    empty_cols = cp.tile(empty_cells_gpu[:, 1], batch_size)

    # Allocate space for results on GPU
    results_gpu = cp.zeros(batch_size, dtype=bool)

    # Allocate space for the batch of boards on GPU
    # We create this once and overwrite it in the loop to save allocation time
    batch_boards_gpu = cp.empty((batch_size, 9, 9), dtype=cp.int8)

    setup_duration = time.time() - start_time_setup
    print(f"GPU Setup time: {setup_duration:.4f} seconds")
    print("\nStarting GPU attempts...")

    # --- Main Loop ---
    total_attempts = 0
    start_time_solve = time.time()
    interval_start_time = start_time_solve
    interval_attempts_count = 0
    interval_batch_count = 0 # Track batches within the interval
    speed_report_interval_sec = 60 # Report speed every 60 seconds
    report_interval_batches = 10 # Report every N batches (keep or adjust)

    try: # Use try/finally to ensure GPU memory is freed if user interrupts
        while True:
            batch_start_time = time.time()

            # 1. Create Batch: Repeat the initial puzzle state B times
            # Efficient way: Expand dims and broadcast/repeat
            batch_boards_gpu[:] = cp.expand_dims(puzzle_gpu, axis=0) # Shape (1, 9, 9) -> broadcasted to (B, 9, 9)

            # 2. Generate Random Fills: Create B * E random numbers (1-9) on GPU
            random_fills = cp.random.randint(1, 10, size=(batch_size * num_empty_cells), dtype=cp.int8)

            # 3. Fill Empty Cells: Use advanced indexing to place random numbers
            batch_boards_gpu[board_indices, empty_rows, empty_cols] = random_fills

            # 4. Check Validity: Launch the CUDA kernel
            # --- TUNING POINT ---
            # Experiment with threads_per_block on your RTX 4080.
            # Powers of 2, multiples of 32. 256, 512, 1024 are common.
            # Start with 512 and see if it's faster than 256.
            threads_per_block = 512 # Try 512 or 1024 instead of 256
            grid_size = math.ceil(batch_size / threads_per_block)
            check_sudoku_batch_kernel((grid_size,), (threads_per_block,), (batch_boards_gpu, results_gpu, batch_size))
            # Kernel launch is asynchronous, but subsequent operations like .any() or .get() will sync.

            # 5. Check for Success: Did *any* board in the batch pass?
            any_success = cp.any(results_gpu) # This syncs the kernel implicitly

            batch_end_time = time.time()
            total_attempts += batch_size
            interval_attempts_count += batch_size # Increment interval counter
            interval_batch_count += 1 # Increment batch counter for interval

            if any_success:
                end_time_solve = time.time()
                print("\n--------------------------------")
                print(f"SUCCESS! Found a solution after ~{total_attempts:,} attempts.")
                print(f"Total solving time: {end_time_solve - start_time_solve:.4f} seconds")

                # Find the index of the first successful board
                # Use cp.where to get all success indices if needed, argmax is fine for first
                success_index = int(cp.argmax(results_gpu).get()) # .get() converts CuPy scalar to Python int

                # Retrieve the solution board from the GPU batch
                solution_board_gpu = batch_boards_gpu[success_index]
                solution_board_np = solution_board_gpu.get() # Transfer to CPU (NumPy)

                print("Solution:")
                print_board(solution_board_np)
                print("--------------------------------")
                return solution_board_np, total_attempts

            # --- Reporting ---
            current_time = time.time()

            # 1. Interval-based Speed Reporting
            elapsed_interval = current_time - interval_start_time
            if elapsed_interval >= speed_report_interval_sec:
                speed = interval_attempts_count / elapsed_interval if elapsed_interval > 0 else 0
                avg_batch_time = elapsed_interval / interval_batch_count if interval_batch_count > 0 else 0

                # Calculate Progress and ETR based on theoretical combinations
                progress_pct = 0.0
                etr_str = "N/A"
                if num_combinations > 0:
                    # Use float for total_attempts in division if num_combinations is huge
                    progress_pct = (float(total_attempts) / num_combinations) * 100.0
                    if speed > 0:
                        remaining_attempts = num_combinations - float(total_attempts)
                        if remaining_attempts > 0:
                            etr_seconds = remaining_attempts / speed
                            # Format ETR nicely
                            etr_delta = datetime.timedelta(seconds=int(etr_seconds))
                            etr_str = str(etr_delta)
                        else:
                            # Already exceeded theoretical, or very close
                            etr_str = "< 1s (theoretically)"
                    else:
                        etr_str = "Infinite (speed is zero)"


                print(f"\n--- Speed Check ({speed_report_interval_sec}s interval) ---")
                print(f"    Batches in interval:  {interval_batch_count}")
                print(f"    Attempts in interval: {interval_attempts_count:,}")
                print(f"    Time elapsed:         {elapsed_interval:.2f}s")
                print(f"    Avg batch time:       {avg_batch_time:.4f}s")
                print(f"    Estimated speed:      {speed:,.2f} attempts/sec")
                print(f"    Total attempts so far: {total_attempts:,}")
                print(f"    Total time elapsed:   {current_time - start_time_solve:.2f}s")
                # Display Progress and ETR
                print(f"    Progress (theoretical): {progress_pct:.6f}%")
                print(f"    Est. Time Remain (thr): {etr_str}")
                print("-----------------------------")
                # Reset for the next interval
                interval_start_time = current_time
                interval_attempts_count = 0
                interval_batch_count = 0

            # 2. Batch-based Progress Reporting (Optional - can be adjusted/removed)
            current_batch_num = total_attempts // batch_size
            if current_batch_num % report_interval_batches == 0 and current_batch_num > 0: # Avoid printing at batch 0
                 elapsed_total = current_time - start_time_solve
                 attempts_per_sec_total = total_attempts / elapsed_total if elapsed_total > 0 else 0
                 # Make this report less verbose if interval report is active
                 print(f"... Batch {current_batch_num}: Checked {total_attempts:,} total boards. "
                       f"Rate (total avg): {attempts_per_sec_total:,.0f} attempts/sec. "
                       f"Last batch time: {batch_end_time - batch_start_time:.4f}s")


            if max_attempts is not None and total_attempts >= max_attempts:
                end_time_solve = time.time()
                print("\n--------------------------------")
                print(f"FAILED: Reached maximum attempts ({max_attempts:,}) without finding a solution.")
                print(f"Total solving time: {end_time_solve - start_time_solve:.4f} seconds")
                print("--------------------------------")
                return None, total_attempts

            # Optional: Brief sleep to allow system responsiveness / prevent 100% CPU on mgmt
            # time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return None, total_attempts
    finally:
        # Clean up GPU memory (optional, but good practice)
        print("Cleaning up GPU memory...")
        del puzzle_gpu, empty_cells_gpu, board_indices, empty_rows, empty_cols
        del results_gpu, batch_boards_gpu, random_fills
        cp.get_default_memory_pool().free_all_blocks()
        print("Cleanup done.")


def solve_sudoku_smart_randomly_gpu(puzzle_board_list, batch_size=1_048_576, max_attempts=None, 
                               use_constraints=True, adaptive_batch=True, 
                               constraint_probability=0.95):
    """
    Enhanced GPU-accelerated random Sudoku solver with smart constraint-based filling.
    
    Args:
        puzzle_board_list: The initial puzzle as a 9x9 Python list of lists.
        batch_size: Initial batch size for parallel board generation.
        max_attempts: Optional limit to total number of attempted boards.
        use_constraints: Whether to use constraint-based filling (True) or pure random (False).
        adaptive_batch: Whether to adapt batch size based on GPU performance.
        constraint_probability: Probability of using constraints vs pure random (0.0-1.0).
    """
    print(f"--- Smart GPU-Accelerated Random Sudoku Solver ---")
    print(f"Initial batch size: {batch_size:,}")
    print(f"Using constraint-based filling: {use_constraints}")
    print(f"Adaptive batch sizing: {adaptive_batch}")
    
    # --- Initialization ---
    start_time_setup = time.time()
    puzzle_board_np = np.array(puzzle_board_list, dtype=np.int8)
    print("Initial Puzzle:")
    print_board(puzzle_board_np)
    
    # Find empty cells and calculate theoretical combinations
    empty_cells_np = np.argwhere(puzzle_board_np == 0)
    num_empty_cells = len(empty_cells_np)
    
    if num_empty_cells == 0:
        print("Board is already full.")
        temp_gpu_board = cp.array(puzzle_board_np).reshape(1, 9, 9)
        temp_results = cp.zeros(1, dtype=bool)
        check_sudoku_batch_kernel((1,), (1,), (temp_gpu_board, temp_results, 1))
        if temp_results.get()[0]:
            print("And it's a valid solution!")
            return puzzle_board_np, 0
        else:
            print("But it's not a valid solution.")
            return None, 0
            
    print(f"\nFound {num_empty_cells} empty cells to fill.")
    
    # Calculate theoretical combinations
    num_combinations = float(9**num_empty_cells)
    print(f"Theoretical combinations: 9^{num_empty_cells} = {num_combinations:,.0f}")
    
    # --- Prepare GPU data ---
    puzzle_gpu = cp.array(puzzle_board_np, dtype=cp.int8)
    empty_cells_gpu = cp.array(empty_cells_np, dtype=cp.int32) # Keep this original reference

    # Function to recalculate batch-dependent GPU arrays
    def recalculate_batch_arrays(current_batch_size, num_empty, empty_cells_gpu_ref):
        board_indices_gpu = cp.repeat(cp.arange(current_batch_size, dtype=cp.int32), num_empty)
        empty_rows_gpu = cp.tile(empty_cells_gpu_ref[:, 0], current_batch_size)
        empty_cols_gpu = cp.tile(empty_cells_gpu_ref[:, 1], current_batch_size)
        results_gpu_alloc = cp.zeros(current_batch_size, dtype=bool)
        batch_boards_gpu_alloc = cp.empty((current_batch_size, 9, 9), dtype=cp.int8)
        return board_indices_gpu, empty_rows_gpu, empty_cols_gpu, results_gpu_alloc, batch_boards_gpu_alloc

    # Initial calculation
    board_indices, empty_rows, empty_cols, results_gpu, batch_boards_gpu = recalculate_batch_arrays(
        batch_size, num_empty_cells, empty_cells_gpu
    )
    
    # --- Analyze puzzle constraints for adaptive behavior ---
    # Check which values are valid for each empty cell
    valid_options_count = []
    if use_constraints:
        print("Analyzing puzzle constraints...")
        for r, c in empty_cells_np:
            # Determine which values (1-9) are already present in this cell's row, column, and 3x3 box
            row_vals = set(puzzle_board_np[r, :])
            col_vals = set(puzzle_board_np[:, c])
            box_r, box_c = 3 * (r // 3), 3 * (c // 3)
            box_vals = set(puzzle_board_np[box_r:box_r+3, box_c:box_c+3].flatten())
            
            # Combine constraints and count valid options
            invalid_vals = row_vals.union(col_vals).union(box_vals)
            valid_vals = [v for v in range(1, 10) if v not in invalid_vals]
            valid_options_count.append(len(valid_vals))
        
        avg_options = sum(valid_options_count) / len(valid_options_count) if valid_options_count else 0
        print(f"Average valid options per empty cell: {avg_options:.2f}")
        
        # Adjust batch size based on constraint analysis if adaptive_batch is enabled
        if adaptive_batch and avg_options < 3.0:
            adjusted_batch = min(batch_size * 4, 2**24)  # Increase but cap at reasonable size
            if adjusted_batch != batch_size: # Check if size actually changes
                print(f"Few options per cell detected. Increasing batch size to {adjusted_batch:,}")
                batch_size = adjusted_batch
                # Recalculate indices and reallocate for new batch size
                board_indices, empty_rows, empty_cols, results_gpu, batch_boards_gpu = recalculate_batch_arrays(
                    batch_size, num_empty_cells, empty_cells_gpu
                )
    
    setup_duration = time.time() - start_time_setup
    print(f"Smart GPU setup time: {setup_duration:.4f} seconds")
    print("\nStarting smart GPU attempts...")
    
    # --- Performance tracking variables ---
    total_attempts = 0
    start_time_solve = time.time()
    interval_start_time = start_time_solve
    interval_attempts_count = 0
    interval_batch_count = 0
    speed_report_interval_sec = 30  # More frequent reporting
    report_interval_batches = 5
    
    # --- Dynamic parameters for constraint-based randomization ---
    use_constraint_fill = use_constraints
    constraint_cycles = 0
    max_constraint_cycles = 10  # How many batches to try with constraints before pure random batch
    random_seed = int(time.time())  # Initial seed for random generation
    
    try:
        while True:
            batch_start_time = time.time()
            
            # 1. Create a batch of puzzle templates
            batch_boards_gpu[:] = cp.expand_dims(puzzle_gpu, axis=0)
            
            # 2. Fill empty cells using either constraint-based or pure random approach
            if use_constraint_fill and constraint_cycles < max_constraint_cycles:
                # Use our custom constraint-based filling kernel
                random_seed = (random_seed * 1664525 + 1013904223) % (2**32)  # Linear congruential generator
                
                # Get current batch size dimensions for kernel launch
                threads_per_block = 512
                block_count = math.ceil((batch_size * num_empty_cells) / threads_per_block)
                
                # Launch smart random fill kernel
                smart_random_fill_kernel((block_count,), (threads_per_block,), 
                                        (batch_boards_gpu, empty_rows, empty_cols, 
                                         board_indices, num_empty_cells, batch_size, random_seed))
                
                constraint_cycles += 1
                if constraint_cycles >= max_constraint_cycles:
                    # Next batch will be pure random
                    constraint_cycles = 0
                    use_constraint_fill = False
                    print("Switching to pure random mode for diversity...")
            else:
                # Use simple random filling (original approach)
                random_fills = cp.random.randint(1, 10, size=(batch_size * num_empty_cells), dtype=cp.int8)
                # Use recalculated arrays for indexing
                batch_boards_gpu[board_indices, empty_rows, empty_cols] = random_fills
                use_constraint_fill = (cp.random.random() < constraint_probability)  # Probabilistic switch
                constraint_cycles = 0
            
            # 3. Check validity of all boards in batch
            threads_per_block = 512
            grid_size = math.ceil(batch_size / threads_per_block)
            check_sudoku_batch_kernel((grid_size,), (threads_per_block,), 
                                     (batch_boards_gpu, results_gpu, batch_size))
            
            # 4. Check for success
            any_success = cp.any(results_gpu)
            
            # 5. Update counts and timing
            batch_end_time = time.time()
            total_attempts += batch_size
            interval_attempts_count += batch_size
            interval_batch_count += 1
            
            if any_success:
                end_time_solve = time.time()
                print("\n==================================")
                print(f"SUCCESS! Found solution after ~{total_attempts:,} attempts.")
                print(f"Total solving time: {end_time_solve - start_time_solve:.4f} seconds")
                
                # Get the successful solution
                success_index = int(cp.argmax(results_gpu).get())
                solution_board_gpu = batch_boards_gpu[success_index]
                solution_board_np = solution_board_gpu.get()
                
                print("Solution:")
                print_board(solution_board_np)
                print("==================================")
                return solution_board_np, total_attempts
            
            # --- Reporting and adaptive tuning ---
            current_time = time.time()
            
            # Speed reporting at intervals
            elapsed_interval = current_time - interval_start_time
            if elapsed_interval >= speed_report_interval_sec:
                speed = interval_attempts_count / elapsed_interval if elapsed_interval > 0 else 0
                avg_batch_time = elapsed_interval / interval_batch_count if interval_batch_count > 0 else 0
                
                # Calculate progress metrics
                progress_pct = (float(total_attempts) / num_combinations) * 100.0 if num_combinations > 0 else 0.0
                etr_str = "N/A"
                if speed > 0 and num_combinations > 0:
                    remaining_attempts = num_combinations - float(total_attempts)
                    if remaining_attempts > 0:
                        etr_seconds = remaining_attempts / speed
                        etr_delta = datetime.timedelta(seconds=int(etr_seconds))
                        etr_str = str(etr_delta)
                    else:
                        etr_str = "< 1s (theoretically)"
                
                print(f"\n--- Performance Update ({speed_report_interval_sec}s interval) ---")
                print(f"    Speed: {speed:,.2f} attempts/sec | Batch time: {avg_batch_time:.4f}s")
                print(f"    Total attempts: {total_attempts:,} | Elapsed: {current_time - start_time_solve:.2f}s")
                print(f"    Progress: {progress_pct:.8f}% | ETR: {etr_str}")
                print(f"    Using constraints: {use_constraint_fill} | Cycle: {constraint_cycles}/{max_constraint_cycles}")
                
                # Adaptive batch size tuning based on performance
                if adaptive_batch and interval_batch_count > 0:
                    target_batch_time = 1.0  # Target seconds per batch
                    if avg_batch_time > 2.0 * target_batch_time:
                        # Too slow, reduce batch size
                        new_batch = max(batch_size // 2, 2**16)
                        if new_batch != batch_size:
                            print(f"    Batches too slow. Reducing batch size: {batch_size:,} → {new_batch:,}")
                            batch_size = new_batch
                            # Reallocate for new batch size
                            board_indices, empty_rows, empty_cols, results_gpu, batch_boards_gpu = recalculate_batch_arrays(
                                batch_size, num_empty_cells, empty_cells_gpu
                            )
                    elif avg_batch_time < 0.5 * target_batch_time:
                        # Too fast, increase batch size
                        new_batch = min(batch_size * 2, 2**24)
                        if new_batch != batch_size:
                            print(f"    Batches too fast. Increasing batch size: {batch_size:,} → {new_batch:,}")
                            batch_size = new_batch
                            # Reallocate for new batch size
                            board_indices, empty_rows, empty_cols, results_gpu, batch_boards_gpu = recalculate_batch_arrays(
                                batch_size, num_empty_cells, empty_cells_gpu
                            )
                
                # Reset interval counters
                interval_start_time = current_time
                interval_attempts_count = 0
                interval_batch_count = 0
            
            # Batch progress reporting
            current_batch_num = total_attempts // batch_size
            if current_batch_num % report_interval_batches == 0 and current_batch_num > 0:
                elapsed_total = current_time - start_time_solve
                rate = total_attempts / elapsed_total if elapsed_total > 0 else 0
                print(f"... Batch {current_batch_num}: {total_attempts:,} boards checked. "
                      f"Rate: {rate:,.0f}/sec. Last batch: {batch_end_time - batch_start_time:.4f}s")
            
            # Check if maximum attempts reached
            if max_attempts is not None and total_attempts >= max_attempts:
                end_time_solve = time.time()
                print("\n==================================")
                print(f"FAILED: Reached maximum {max_attempts:,} attempts without finding a solution.")
                print(f"Total solving time: {end_time_solve - start_time_solve:.4f} seconds")
                print("==================================")
                return None, total_attempts
                
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return None, total_attempts
    finally:
        # Clean up GPU memory
        print("Cleaning up GPU memory...")
        del puzzle_gpu, empty_cells_gpu, board_indices, empty_rows, empty_cols
        del results_gpu, batch_boards_gpu
        cp.get_default_memory_pool().free_all_blocks()
        print("Cleanup done.")


def solve_sudoku_advanced_gpu(puzzle_board_list, batch_size=4_194_304, max_attempts=None, 
                           difficulty_bias=3.0, max_attempts_per_batch=None):
    """
    Advanced GPU-accelerated Sudoku solver using optimized kernels and heuristics.
    
    Args:
        puzzle_board_list: The initial puzzle as a 9x9 Python list of lists.
        batch_size: Initial batch size for parallel board generation.
        max_attempts: Optional limit to total number of attempted boards.
        difficulty_bias: Bias factor for difficult cells (1.0-10.0).
        max_attempts_per_batch: Max attempts per batch before adjusting strategy.
    """
    print(f"--- Advanced GPU-Accelerated Sudoku Solver ---")
    print(f"Initial batch size: {batch_size:,}")
    print(f"Difficulty bias: {difficulty_bias:.1f}")
    
    # --- Initialization ---
    start_time_setup = time.time()
    puzzle_board_np = np.array(puzzle_board_list, dtype=np.int8)
    print("Initial Puzzle:")
    print_board(puzzle_board_np)
    
    # Find empty cells
    empty_cells_np = np.argwhere(puzzle_board_np == 0)
    num_empty_cells = len(empty_cells_np)
    
    if num_empty_cells == 0:
        print("Board is already full.")
        temp_gpu_board = cp.array(puzzle_board_np).reshape(1, 9, 9)
        temp_results = cp.zeros(1, dtype=bool)
        check_sudoku_batch_kernel((1,), (1,), (temp_gpu_board, temp_results, 1))
        if temp_results.get()[0]:
            print("And it's a valid solution!")
            return puzzle_board_np, 0
        else:
            print("But it's not a valid solution.")
            return None, 0
            
    print(f"\nFound {num_empty_cells} empty cells to fill.")
    
    # Calculate theoretical combinations
    num_combinations = float(9**num_empty_cells)
    print(f"Theoretical combinations: 9^{num_empty_cells} = {num_combinations:,.0f}")
    
    # --- Prepare GPU data ---
    puzzle_gpu = cp.array(puzzle_board_np, dtype=cp.int8)
    empty_cells_gpu = cp.array(empty_cells_np, dtype=cp.int32)

    # Prepare batch arrays
    board_indices = cp.repeat(cp.arange(batch_size, dtype=cp.int32), num_empty_cells)
    empty_rows = cp.tile(empty_cells_gpu[:, 0], batch_size)
    empty_cols = cp.tile(empty_cells_gpu[:, 1], batch_size)
    results_gpu = cp.zeros(batch_size, dtype=bool)
    batch_boards_gpu = cp.empty((batch_size, 9, 9), dtype=cp.int8)
    
    # --- Analyze puzzle constraints ---
    print("Analyzing puzzle difficulty...")
    cell_difficulties = []
    
    for r, c in empty_cells_np:
        # Count valid options for this cell
        invalid_vals = set()
        
        # Row constraints
        for col in range(9):
            val = puzzle_board_np[r, col]
            if val > 0:
                invalid_vals.add(val)
                
        # Column constraints
        for row in range(9):
            val = puzzle_board_np[row, c]
            if val > 0:
                invalid_vals.add(val)
                
        # Box constraints
        box_r, box_c = 3 * (r // 3), 3 * (c // 3)
        for row in range(box_r, box_r + 3):
            for col in range(box_c, box_c + 3):
                val = puzzle_board_np[row, col]
                if val > 0:
                    invalid_vals.add(val)
                    
        # Calculate difficulty based on remaining options
        valid_options = 9 - len(invalid_vals)
        difficulty = 10.0 / valid_options if valid_options > 0 else 10.0
        cell_difficulties.append(difficulty)
    
    avg_difficulty = sum(cell_difficulties) / len(cell_difficulties) if cell_difficulties else 1.0
    print(f"Average cell difficulty: {avg_difficulty:.2f} (higher means fewer options)")
    
    # Adjust batch size based on difficulty
    adaptive_size = batch_size
    if avg_difficulty > 4.0:
        # For very difficult puzzles, use larger batches
        adaptive_size = min(batch_size * 4, 2**24)
        print(f"High difficulty detected. Increasing batch size to {adaptive_size:,}")
        
        # Recalculate indices for new batch size
        if adaptive_size != batch_size:
            batch_size = adaptive_size
            board_indices = cp.repeat(cp.arange(batch_size, dtype=cp.int32), num_empty_cells)
            empty_rows = cp.tile(empty_cells_gpu[:, 0], batch_size)
            empty_cols = cp.tile(empty_cells_gpu[:, 1], batch_size)
            results_gpu = cp.zeros(batch_size, dtype=bool)
            batch_boards_gpu = cp.empty((batch_size, 9, 9), dtype=cp.int8)
    
    # Set solver parameters
    if max_attempts_per_batch is None:
        max_attempts_per_batch = batch_size * 20  # Default: try each approach for 20 batches
    
    setup_duration = time.time() - start_time_setup
    print(f"Advanced setup time: {setup_duration:.4f} seconds")
    print("\nStarting advanced GPU solving...")
    
    # --- Performance tracking variables ---
    total_attempts = 0
    start_time_solve = time.time()
    last_report_time = start_time_solve
    last_report_attempts = 0
    report_interval = 5  # seconds
    
    # --- Solver strategy variables ---
    random_seed = int(time.time())
    current_bias = difficulty_bias
    constraint_mode = True  # Start with constraint-based filling
    strategy_attempts = 0   # Attempts with current strategy
    strategy_switches = 0   # Count strategy switches
    
    # --- Main solving loop ---
    try:
        while True:
            batch_start_time = time.time()
            
            # Create batch of puzzle templates
            batch_boards_gpu[:] = cp.expand_dims(puzzle_gpu, axis=0)
            
            # Update random seed
            random_seed = (random_seed * 1664525 + 1013904223) % (2**32)
            
            # Fill empty cells using the current strategy
            threads_per_block = 512
            block_count = math.ceil((batch_size * num_empty_cells) / threads_per_block)
            
            if constraint_mode:
                # Use the advanced smart fill kernel with current bias
                advanced_smart_fill_kernel((block_count,), (threads_per_block,), 
                                          (batch_boards_gpu, empty_rows, empty_cols, 
                                           board_indices, num_empty_cells, batch_size, 
                                           random_seed, current_bias))
            else:
                # Use simple random filling without constraints
                random_fills = cp.random.randint(1, 10, size=(batch_size * num_empty_cells), dtype=cp.int8)
                batch_boards_gpu[board_indices, empty_rows, empty_cols] = random_fills
            
            # Check validity
            grid_size = math.ceil(batch_size / threads_per_block)
            check_sudoku_batch_kernel((grid_size,), (threads_per_block,), 
                                     (batch_boards_gpu, results_gpu, batch_size))
            
            # Check for success
            any_success = cp.any(results_gpu)
            
            # Update counts and timing
            batch_end_time = time.time()
            total_attempts += batch_size
            strategy_attempts += batch_size
            
            if any_success:
                end_time_solve = time.time()
                print("\n==================================")
                print(f"SUCCESS! Found solution after ~{total_attempts:,} attempts.")
                print(f"Total solving time: {end_time_solve - start_time_solve:.4f} seconds")
                
                # Get the successful solution
                success_index = int(cp.argmax(results_gpu).get())
                solution_board_gpu = batch_boards_gpu[success_index]
                solution_board_np = solution_board_gpu.get()
                
                print("Solution:")
                print_board(solution_board_np)
                print("==================================")
                return solution_board_np, total_attempts
            
            # --- Adaptive strategy adjustments ---
            current_time = time.time()
            
            # Check if it's time to switch strategies
            if strategy_attempts >= max_attempts_per_batch:
                # Switch between constraint-based and pure random
                constraint_mode = not constraint_mode
                strategy_attempts = 0
                strategy_switches += 1
                
                # Adjust difficulty bias
                if constraint_mode:
                    # When switching back to constraint mode, adjust bias
                    if strategy_switches % 3 == 0:
                        current_bias = max(1.0, current_bias - 0.5)  # Decrease bias
                    elif strategy_switches % 5 == 0:
                        current_bias = min(10.0, current_bias + 1.0)  # Increase bias
                
                print(f"Strategy switch #{strategy_switches}: "
                      f"{'Using constraints' if constraint_mode else 'Using pure random'} "
                      f"(bias: {current_bias:.1f})")
            
            # Periodic reporting
            if current_time - last_report_time >= report_interval:
                elapsed = current_time - start_time_solve
                attempts_since_last = total_attempts - last_report_attempts
                rate = attempts_since_last / (current_time - last_report_time)
                total_rate = total_attempts / elapsed if elapsed > 0 else 0
                
                print(f"\nProgress update: {total_attempts:,} attempts in {elapsed:.2f}s")
                print(f"Current rate: {rate:,.0f}/sec | Overall rate: {total_rate:,.0f}/sec")
                print(f"Strategy: {'Constraint-based' if constraint_mode else 'Pure random'} | "
                      f"Bias: {current_bias:.1f} | Switches: {strategy_switches}")
                
                # Reset reporting counters
                last_report_time = current_time
                last_report_attempts = total_attempts
            
            # Check if maximum attempts reached
            if max_attempts is not None and total_attempts >= max_attempts:
                end_time_solve = time.time()
                print("\n==================================")
                print(f"FAILED: Reached maximum {max_attempts:,} attempts without finding a solution.")
                print(f"Total solving time: {end_time_solve - start_time_solve:.4f} seconds")
                print("==================================")
                return None, total_attempts
                
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return None, total_attempts
    finally:
        # Clean up GPU memory
        print("Cleaning up GPU memory...")
        del puzzle_gpu, empty_cells_gpu, board_indices, empty_rows, empty_cols
        del results_gpu, batch_boards_gpu
        cp.get_default_memory_pool().free_all_blocks()
        print("Cleanup done.")


# --- Testing and Benchmarking Framework ---

def generate_test_puzzles(difficulty_levels=3, puzzles_per_level=2):
    """
    Generate a set of test puzzles with various difficulty levels.
    
    Args:
        difficulty_levels: Number of different difficulty levels to generate
        puzzles_per_level: Number of puzzles per difficulty level
    
    Returns:
        Dictionary mapping difficulty levels to lists of puzzles
    """
    # Pre-made puzzles of varying difficulty (number of empty cells)
    easy_puzzles = [
        # 10-15 empty cells, well-distributed
        [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 0, 3, 4, 8], # 1 empty
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 0, 6, 1, 0, 2, 3], # 2 empty -> 3 total
            [4, 2, 6, 8, 5, 3, 7, 9, 1], # Made solvable
            [7, 1, 3, 9, 2, 0, 8, 5, 6], # 1 empty -> 4 total
            [9, 6, 1, 5, 3, 7, 2, 8, 4], # Made solvable
            [2, 8, 7, 4, 1, 9, 6, 3, 5], # Made solvable
            [3, 4, 5, 2, 8, 6, 1, 7, 9]  # Made solvable
        ],
        # Another easier puzzle
        [
            [0, 0, 0, 2, 6, 0, 7, 0, 1],
            [6, 8, 0, 0, 7, 0, 0, 9, 0],
            [1, 9, 0, 0, 0, 4, 5, 0, 0],
            [8, 2, 0, 1, 0, 0, 0, 4, 0],
            [0, 0, 4, 6, 0, 2, 9, 0, 0],
            [0, 5, 0, 0, 0, 3, 0, 2, 8],
            [0, 0, 9, 3, 0, 0, 0, 7, 4],
            [0, 4, 0, 0, 5, 0, 0, 3, 6],
            [7, 0, 3, 0, 1, 8, 0, 0, 0]
        ]
    ]
    
    medium_puzzles = [
        # 25-35 empty cells
        [
            [0, 2, 0, 6, 0, 8, 0, 0, 0],
            [5, 8, 0, 0, 0, 9, 7, 0, 0],
            [0, 0, 0, 0, 4, 0, 0, 0, 0],
            [3, 7, 0, 0, 0, 0, 5, 0, 0],
            [6, 0, 0, 0, 0, 0, 0, 0, 4],
            [0, 0, 8, 0, 0, 0, 0, 1, 3],
            [0, 0, 0, 0, 2, 0, 0, 0, 0],
            [0, 0, 9, 8, 0, 0, 0, 3, 6],
            [0, 0, 0, 3, 0, 6, 0, 9, 0]
        ],
        # Another medium puzzle
        [
            [1, 0, 0, 4, 8, 9, 0, 0, 6],
            [7, 3, 0, 0, 0, 0, 0, 4, 0],
            [0, 0, 0, 0, 0, 1, 2, 9, 5],
            [0, 0, 7, 1, 2, 0, 6, 0, 0],
            [5, 0, 0, 7, 0, 3, 0, 0, 8],
            [0, 0, 6, 0, 9, 5, 7, 0, 0],
            [9, 1, 4, 6, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 3, 7],
            [8, 0, 0, 5, 1, 2, 0, 0, 4]
        ]
    ]
    
    hard_puzzles = [
        # 45-55 empty cells
        [
            [0, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 6, 0, 0, 0, 0, 3],
            [0, 7, 4, 0, 8, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 0, 0, 2],
            [0, 8, 0, 0, 4, 0, 0, 1, 0],
            [6, 0, 0, 5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 7, 8, 0],
            [5, 0, 0, 0, 0, 9, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 4, 0]
        ],
        # Another hard puzzle - the "hardest" Sudoku according to some sources
        [
            [8, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 6, 0, 0, 0, 0, 0],
            [0, 7, 0, 0, 9, 0, 2, 0, 0],
            [0, 5, 0, 0, 0, 7, 0, 0, 0],
            [0, 0, 0, 0, 4, 5, 7, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 3, 0],
            [0, 0, 1, 0, 0, 0, 0, 6, 8],
            [0, 0, 8, 5, 0, 0, 0, 1, 0],
            [0, 9, 0, 0, 0, 0, 4, 0, 0]
        ]
    ]
    
    # Prepare the test set with the appropriate mix of difficulty levels
    test_puzzles = {
        "easy": easy_puzzles,
        "medium": medium_puzzles,
        "hard": hard_puzzles
    }
    
    return test_puzzles


def benchmark_solvers(test_puzzles, batch_sizes=None, max_attempts_per_puzzle=10**8):
    """
    Benchmark different solver variations against test puzzles.
    
    Args:
        test_puzzles: Dictionary of puzzles by difficulty level
        batch_sizes: List of batch sizes to try (powers of 2 recommended)
        max_attempts_per_puzzle: Maximum attempts to make before giving up on a puzzle
    """
    if batch_sizes is None:
        batch_sizes = [2**20, 2**22]  # Default batch sizes to test
    
    results = {}
    
    print("\n" + "="*50)
    print("SUDOKU SOLVER BENCHMARK")
    print("="*50)
    
    # Track overall stats
    total_puzzles = sum(len(puzzles) for puzzles in test_puzzles.values())
    puzzles_completed = 0
    
    # Run each algorithm variant against all test puzzles
    for difficulty, puzzles in test_puzzles.items():
        results[difficulty] = {}
        
        print(f"\n--- Testing {len(puzzles)} {difficulty} puzzles ---")
        
        for i, puzzle in enumerate(puzzles):
            puzzle_results = {}
            print(f"\nPuzzle {i+1}/{len(puzzles)} ({difficulty}):")
            print_board(np.array(puzzle, dtype=np.int8))
            empty_count = sum(row.count(0) for row in puzzle)
            print(f"Empty cells: {empty_count}")
            
            # Test original random solver
            for batch_size in batch_sizes:
                print(f"\n* Testing original random solver with batch size {batch_size:,}")
                start_time = time.time()
                solution, attempts = solve_sudoku_randomly_gpu(
                    puzzle,
                    batch_size=batch_size,
                    max_attempts=max_attempts_per_puzzle
                )
                end_time = time.time()
                duration = end_time - start_time
                
                result = {
                    "solved": solution is not None,
                    "attempts": attempts,
                    "time": duration,
                    "speed": attempts / duration if duration > 0 else 0
                }
                
                if solution is not None:
                    print(f"✓ Solved in {duration:.2f} seconds after {attempts:,} attempts")
                    print(f"  Speed: {result['speed']:,.2f} attempts/sec")
                else:
                    print(f"✗ Failed after {duration:.2f} seconds and {attempts:,} attempts")
                
                puzzle_results[f"original_{batch_size}"] = result
            
            # Test smart random solver
            for batch_size in batch_sizes:
                print(f"\n* Testing smart constraint solver with batch size {batch_size:,}")
                start_time = time.time()
                solution, attempts = solve_sudoku_smart_randomly_gpu(
                    puzzle,
                    batch_size=batch_size,
                    max_attempts=max_attempts_per_puzzle,
                    use_constraints=True,
                    adaptive_batch=True
                )
                end_time = time.time()
                duration = end_time - start_time
                
                result = {
                    "solved": solution is not None,
                    "attempts": attempts,
                    "time": duration,
                    "speed": attempts / duration if duration > 0 else 0
                }
                
                if solution is not None:
                    print(f"✓ Solved in {duration:.2f} seconds after {attempts:,} attempts")
                    print(f"  Speed: {result['speed']:,.2f} attempts/sec")
                else:
                    print(f"✗ Failed after {duration:.2f} seconds and {attempts:,} attempts")
                
                puzzle_results[f"smart_{batch_size}"] = result
            
            results[difficulty][f"puzzle_{i}"] = puzzle_results
            
            # Update progress
            puzzles_completed += 1
            print(f"\nProgress: {puzzles_completed}/{total_puzzles} puzzles tested")
        
    # Print summary report
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    
    for difficulty in results:
        print(f"\n--- {difficulty.upper()} PUZZLES SUMMARY ---")
        
        # Collect stats for this difficulty level
        solvers = {}
        for puzzle_data in results[difficulty].values():
            for solver_name, solver_result in puzzle_data.items():
                if solver_name not in solvers:
                    solvers[solver_name] = {
                        "total_solved": 0,
                        "total_time": 0,
                        "total_attempts": 0,
                        "count": 0
                    }
                
                solvers[solver_name]["count"] += 1
                if solver_result["solved"]:
                    solvers[solver_name]["total_solved"] += 1
                solvers[solver_name]["total_time"] += solver_result["time"]
                solvers[solver_name]["total_attempts"] += solver_result["attempts"]
        
        # Display stats for each solver on this difficulty
        for solver_name, stats in solvers.items():
            success_rate = (stats["total_solved"] / stats["count"]) * 100
            avg_time = stats["total_time"] / stats["count"]
            avg_attempts = stats["total_attempts"] / stats["count"]
            avg_speed = stats["total_attempts"] / stats["total_time"] if stats["total_time"] > 0 else 0
            
            print(f"\n{solver_name}:")
            print(f"  Success rate: {success_rate:.1f}% ({stats['total_solved']}/{stats['count']})")
            print(f"  Avg. time per puzzle: {avg_time:.2f} seconds")
            print(f"  Avg. attempts per puzzle: {avg_attempts:,.0f}")
            print(f"  Avg. speed: {avg_speed:,.2f} attempts/sec")
    
    return results


# --- Run full testing when executed directly ---
if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='CUDA-accelerated Sudoku solver')
    parser.add_argument('--mode', type=int, choices=[1, 2], default=1,
                        help='Test mode: 1=Quick Test (single puzzle), 2=Full Benchmark (multiple puzzles)')
    parser.add_argument('--batch-size', type=int, default=2**22,
                        help='Batch size for GPU processing (default: 2^22 = 4,194,304)')
    parser.add_argument('--max-attempts', type=int, default=10**8,
                        help='Maximum attempts before giving up (default: 10^8 = 100,000,000)')
    parser.add_argument('--solver', type=str, choices=['original', 'smart', 'advanced', 'all'], default='smart',
                        help='Solver algorithm to use: original, smart, advanced, or all (default: smart)')
    parser.add_argument('--difficulty-bias', type=float, default=3.0,
                        help='Difficulty bias for advanced solver (1.0-10.0, default: 3.0)')
    parser.add_argument('--adaptive', action='store_true',
                        help='Use adaptive batch sizing for smart solver')
    parser.add_argument('--puzzle-difficulty', type=str, choices=['easy', 'medium', 'hard'], default='easy',
                        help='Difficulty level for quick test mode (default: easy)')
    parser.add_argument('--puzzle-index', type=int, default=0,
                        help='Puzzle index to use for quick test (default: 0)')
    
    args = parser.parse_args()
    
    # Set up environment for benchmarking
    print("\nCUDA TEST ENVIRONMENT")
    print(f"GPU Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')}")
    print(f"CUDA Version: {cp.cuda.runtime.driverGetVersion()}")
    print(f"CuPy Version: {cp.__version__}")
    print(f"CPU Cores: {multiprocessing.cpu_count()}")
    print(f"Batch size: {args.batch_size:,}")
    print(f"Max attempts: {args.max_attempts:,}")
    
    # Test mode based on command line arguments
    if args.mode == 1:
        # Quick test with a single puzzle
        print(f"\nRunning quick test with {args.puzzle_difficulty} puzzle (index {args.puzzle_index})...")
        
        # Get puzzles for the selected difficulty
        puzzles = generate_test_puzzles()[args.puzzle_difficulty]
        
        # Make sure the requested puzzle index exists
        if args.puzzle_index >= len(puzzles):
            print(f"Error: Puzzle index {args.puzzle_index} out of range for {args.puzzle_difficulty} difficulty.")
            print(f"Using index 0 instead (valid indices: 0-{len(puzzles)-1}).")
            args.puzzle_index = 0
        
        initial_puzzle = puzzles[args.puzzle_index]
        
        # Run the selected solver(s)
        if args.solver in ['original', 'both']:
            print("\n--- Testing Original Solver ---")
            solution, attempts = solve_sudoku_randomly_gpu(
                initial_puzzle,
                batch_size=args.batch_size,
                max_attempts=args.max_attempts
            )
        
        if args.solver in ['smart', 'both']:
            print("\n--- Testing Smart Solver ---")
            solution, attempts = solve_sudoku_smart_randomly_gpu(
                initial_puzzle,
                batch_size=args.batch_size,
                max_attempts=args.max_attempts,
                use_constraints=True,
                adaptive_batch=args.adaptive
            )
        
        if args.solver in ['advanced', 'both']:
            print("\n--- Testing Advanced Solver ---")
            solution, attempts = solve_sudoku_advanced_gpu(
                initial_puzzle,
                batch_size=args.batch_size,
                max_attempts=args.max_attempts,
                difficulty_bias=args.difficulty_bias
            )
    
    elif args.mode == 2:
        # Run full benchmark
        print("\nRunning full benchmark...")
        test_puzzles = generate_test_puzzles(difficulty_levels=3, puzzles_per_level=2)
        batch_sizes = [args.batch_size]
        benchmark_results = benchmark_solvers(test_puzzles, batch_sizes, max_attempts_per_puzzle=args.max_attempts)
    
    print("\nTesting complete!")
