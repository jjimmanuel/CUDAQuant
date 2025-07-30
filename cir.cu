# include <curand_kernel.h>
# include <vector>
# include <iostream>


const int threads_per_block = 256;

void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(ans) { checkCudaError((ans), __FILE__, __LINE__); }



__global__ void setupCurand(curandState *states, unsigned long seed, int n) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < n)
		curand_init(seed, idx, 0, &states[idx]);
}


__global__ void cir(curandState *states, float *results, float kappa, float theta, 
			float sigma, float r0, float dt, int numPaths, int numSteps) {	
	
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= numPaths) return; 

	curandState localState = states[idx];

	float r = r0;
	results[idx * (numSteps + 1)] = r;

	// The non-central chi squared distribution can be simulated using the a poisson random variable with a specific mean and a gamma random variable
	for (int i = 0; i < numSteps; ++i) {
		float c = (sigma * sigma * (1.0 - exp(-kappa * dt))) / (4.0 * kappa);
		float d = (4.0 * kappa * theta) / (sigma * sigma);
		float lambda = (4.0 * kappa * exp(-kappa * dt) * r) / (sigma * sigma * (1.0 - exp(-kappa * dt)));
		float X;
		int n_poisson = curand_poisson(&localState, lambda / 2.0);
		float central_chi_df = d + 2.0 * n_poisson;
		float X;
		if (central_chi_df <= 0.0) {
			X = 0.0;
		} else {
			X = curand_gamma(&localState, central_chi_df / 2.0, 2.0
		}

		r = c * X;
		results[idx * (numSteps + 1) + (i + 1)] = r;
	}

	states[idx] = localState;

}

int main() {

	// 1. Simulation parameters
	const int numPaths = 10000;
	const int numSteps = 252;
	const float kappa = 3;
	const float theta = 2;
	const float sigma = 1;
	const float r0 = 0.03;
	const int T = 1.0;
	const float dt = T / numSteps

	// 2. Allocate device and host memory
	float *device_results;
	curandState *device_states;

	std::vector<float> host_results(numPaths * (numSteps + 1));
	
	CUDA_CHECK(cudaMalloc(&device_results, numPaths * (numSteps + 1) * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&device_states, numPaths * sizeof(curandState)));

	// 3. Setup curand states
	int blocks = (numPaths + threads_per_block - 1) / threads_per_block;
	setupCurand<<<blocks, threads_per_block>>>(device_states, 0, numPaths);
	CUDA_CHECK(cudaDeviceSynchronize());

	// 4. Launch Cox Ingersoll Ross kernel
	cir<<<blocks, threads_per_block>>>(device_states, device_results, kappa, theta, sigma, 
						r0, dt, numPaths, numSteps);
	

	CUDA_CHECK(cudaDeviceSynchronize());

	// 5. Transfer results from DEvice to Host
	CUDA_CHECK(cudaMemcpy(host_results.data(), device_results, numPaths * (numSteps + 1) * sizeof(float), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(device_results));
	CUDA_CHECK(cudaFree(device_states));

	return 0;

}
	
	
