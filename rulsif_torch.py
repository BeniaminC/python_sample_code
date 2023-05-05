"""
RuLSIF Time-Series Anomaly detection using torch
Author: Beniamin Condrea
Written 7/18/2022
"""
import sys

sys.path.append("..")
sys.path.append(".")


from typing import Any, Callable, Dict

import torch
from torch.backends import cuda


torch.set_printoptions(precision=5)
torch.set_default_dtype(torch.float32)
cuda.matmul.allow_tf32 = True


def tensor_debugging(tensor: torch.Tensor) -> None:
    """
    Method for printing/logging tensor information, generally for debugging. Prints the the shape, datatype, and device of the tensor.

        Args:
            tensor (:obj:`torch.Tensor`): torch.Tensor object

        Returns:
            :obj:`set[tuple[int], torch.dtype, tensor.device]`

        Example::

            t = torch.Tensor([1,2,3,4,5])
            tensor_debugging(t)
    """
    return tensor.shape, tensor.dtype, tensor.device


def cuda_debugging(device: str) -> Dict[str, Any]:
    """
    Method for printing complete cuda information. Prints the current device, device count, architecture list,
    device capacity, device name, device properties, memory usage, utilization, memory information, memory statistics, and memory
    summary.

        Args:

            device (:obj:`torch.device` or :obj:`str`): cuda device

        Returns:
            :obj:`dict`
        Example::

            cuda_debugging('cuda:0')
    """
    assert "cuda" in device
    current_device = torch.cuda.current_device()
    device_count = torch.cuda.device_count()
    arch_list = torch.cuda.get_arch_list()
    device_cap = torch.cuda.get_device_capability(device)
    device_name = torch.cuda.get_device_name(device)
    device_prop = torch.cuda.get_device_properties(device)
    memory_usage = torch.cuda.memory_usage(device)
    utilization = torch.cuda.utilization(device)
    mem_info = torch.cuda.mem_get_info()
    memory_stats = torch.cuda.memory_stats()
    memory_summary = torch.cuda.memory_summary()
    cuda_info = {
        "current device": current_device,
        "device count": device_count,
        "arch list": arch_list,
        "device capability": device_cap,
        "device name": device_name,
        "device properties": device_prop,
        "memory usage": memory_usage,
        "utilization": utilization,
        "memory info": mem_info,
        "memory stats": memory_stats,
        "memory summary": memory_summary,
    }
    return cuda_info


class RulsifTimeSeriesAnomalyDetection:
    """
    Apply RuLSIF anomaly detection on multi-dimensional time series data.

        Args:
            data (:obj:`torch.Tensor`): time-series data. Support multidimensional data.
            sample_width (:obj:`int`): the number of datapoints per sample. This is the
                k value. Increasing this value will use more memory and increase
                computation time. Defaults to 10.
            retro_width (:obj:`int`): the distance between the sample comparisions. This
                also increases the sampling size. Increasing this value will dra-
                matically use more memory dramatically increase computation time.
                Defaults to 50.
            sigma (:obj:`float`): the sigma value of the data. It is advised to first
                compute the sigma value of the data before setting this value.
                Defaults to 1.0.
            alpha (:obj:`float`): ratio of comparisons. The forward/backward relative
                density-ratio. If `alpha = 0`, the relative density-ratio is re-
                duced to plain density-ratio.
            _lambda (:obj:`float`): regularization parameter. This is used to prevent over-
                fitting. Higher values will reduce overfitting, but too high will cause underfitting.
                device (:obj:`torch.device` or :obj:`str`): sets the computation to perform on the CPU
                or the GPU supported by CUDA. Note that the data remains on the host.
                This is generally set to `1, 0.1, 0.01, 0.001`
            batch_size(:obj:`int`): sets the batch size to perform vectorized computation. Note this
                value should be set to binary value (i.e., 2,4,8,16,...). Typically, a GPU's warp size is
                32 or more. Uses more memory and computation!

        Returns:
            :obj:`None`

        Example::

            n = torch.normal(mean=0., std=1., size=(20,2)) # 2-dimensional data
            r= RulsifTimeSeriesAnomalyDetection(n, 10, 50, 0.569, 0.1, 0.01, 'cuda', 2048)
    """

    def __init__(
        self,
        data: torch.Tensor = torch.tensor(()),
        sample_width: int | torch.Tensor = 10,
        retro_width: int | torch.Tensor = 50,
        sigma: float | torch.Tensor = 1.0,
        alpha: float | torch.Tensor = 0.1,
        _lambda: float | torch.Tensor = 0.01,
        device: str = "cpu",
        batch_size: int | torch.Tensor = 32
    ):
        self.__device = device
        self.__data = data.to(device=self.__device)

        self.__sample_width = torch.tensor(
            sample_width, dtype=torch.int32, device=self.__device
        )
        self.__retro_width = torch.tensor(
            retro_width, dtype=torch.int32, device=self.__device
        )
        self.__sigma = torch.tensor(sigma, dtype=torch.float32, device=self.__device)
        self.__alpha = torch.tensor(alpha, dtype=torch.float32, device=self.__device)
        self.__lambda = torch.tensor(_lambda, dtype=torch.float32, device=self.__device)
        self.__batch_size = torch.tensor(
            batch_size, dtype=torch.int32, device=self.__device
        )
        self.__dissimilarities_size = torch.tensor(
            self.__data.shape[0] - self.__sample_width - (2 * self.__retro_width) + 2,
            dtype=torch.int32,
            device=self.__device,
        )
        self.__dissimilarities = None
        self.__batch_dissimilarities = None

    def set_dissimilarities(self) -> None:
        """
        Sets the dissimilarities without batching. May not effectively use the GPU. Will perform more slowly than
        `set_batch_dissimilarities()`.

        Example::

            n = torch.normal(mean=0., std=1., size=(20,2)) # 2-dimensional data
            r = RulsifTimeSeriesAnomalyDetection(n, 10, 50, 0.569, 0.1, 0.01, 'cuda', 2048)
            r.set_dissimilarities() # sets the dissimilarities to RuLSIF dissimilarities
        """
        try:
            assert (
                self.__dissimilarities_size > 0
            ), f"""Cannot run Rulsif algorithm under these conditions.
                Dissimilarity size: {self.__dissimilarities_size}.
                Input more data, or decrease the sample width and/or retro width"""
        except AssertionError as err:
            raise err
        self.__dissimilarities = torch.zeros(
            (self.__dissimilarities_size, 4), dtype=torch.float32, device=self.__device
        )
        sigma_gaussian_kernel_d2 = RulsifTimeSeriesAnomalyDetection.generate_torch_k_gaussian_kernel(
            self.__sigma, dim=2
        )
        subsequences = RulsifTimeSeriesAnomalyDetection.convert_data_to_windows(
            self.__data, self.__sample_width
        )
        try:
            for idx in [
                (x, x + self.__retro_width, x + (self.__retro_width * 2))
                for x in range(0, self.__dissimilarities_size, 1)
            ]:
                y_1 = subsequences[idx[0] : idx[1]]
                y_2 = subsequences[idx[1] : idx[2]]
                diss_calc = self.bidirectional_dissimilarity(
                    y_1, y_2, self.__alpha, self.__lambda, sigma_gaussian_kernel_d2
                )
                tensor_debugging(diss_calc)
                point_x = torch.tensor(
                    idx[1] + self.__sample_width // 2,
                    dtype=torch.float32,
                    device=self.__device,
                )
                diss_data = torch.tensor(
                    (point_x, diss_calc[0], diss_calc[1], diss_calc[2]),
                    device=self.__device,
                )
                self.__dissimilarities[idx[0]] = diss_data
        except RuntimeError as err:
            torch.cuda.ipc_collect()
            raise err

    def set_batch_dissimilarities(self) -> None:
        """
        Sets the dissimilarities with batching. May more effectively use the GPU. Will perform faster than `set_dissimilarities()`.
        Larger batch size will generally improve GPU utilization. Will throw `RuntimeError: CUDA out of memory.` if batch size is too large.

        Example::

            n = torch.normal(mean=0., std=1., size=(20,2)) # 2-dimensional data
            r = RulsifTimeSeriesAnomalyDetection(n, 10, 50, 0.569, 0.1, 0.01, 'cuda', 2048)
            r.set_batch_dissimilarities() # sets the dissimilarities to RuLSIF dissimilarities
        """
        try:
            assert (
                self.__dissimilarities_size > 0
            ), f"""Cannot run Rulsif algorithm under these conditions.
                Dissimilarity size: {self.__dissimilarities_size}.
                Input more data, or decrease the sample width and/or retro width"""
        except AssertionError as err:
            raise err
        self.__batch_dissimilarities = torch.zeros(
            (self.__dissimilarities_size, 4), device=self.__device
        )
        sigma_gaussian_kernel_d3 = self.generate_torch_k_gaussian_kernel(
            self.__sigma, dim=3
        )
        batch_subsequences = (
            RulsifTimeSeriesAnomalyDetection.convert_data_to_windows_batches(
                self.__data, self.__sample_width, self.__retro_width
            )
        )
        offset = self.get_offset()
        assert (
            self.__batch_size > 0
        ), f"Batch size must be at least 1. Current batch size: {self.__batch_size}"
        try:
            start_idx = torch.tensor(0, dtype=torch.int32, device=self.__device)
            end_idx = torch.tensor(self.__batch_size, device=self.__device)
            y_1_batch_subsequences = batch_subsequences[: self.__dissimilarities_size]
            y_2_batch_subsequences = batch_subsequences[self.__retro_width :]
            assert y_1_batch_subsequences.shape[0] == y_2_batch_subsequences.shape[0]
            while start_idx < self.__dissimilarities_size:
                y_1 = y_1_batch_subsequences[start_idx:end_idx]
                y_2 = y_2_batch_subsequences[start_idx:end_idx]
                diss_calc = self.fast_bidirectional_dissimilarity2(
                    y_1, y_2, self.__alpha, self.__lambda, sigma_gaussian_kernel_d3
                )
                idx_offset = start_idx + offset
                point_x = torch.arange(
                    idx_offset,
                    y_1.shape[0] + idx_offset,
                    1,
                    dtype=torch.float32,
                    device=self.__device,
                )
                diss_data = torch.column_stack(
                    (point_x, diss_calc[0], diss_calc[1], diss_calc[2])
                )
                self.__batch_dissimilarities[start_idx:end_idx] = diss_data
                start_idx += self.__batch_size
                end_idx += self.__batch_size
        except RuntimeError as err:
            torch.cuda.ipc_collect()
            raise err

    def get_dissimilarities(self) -> torch.Tensor:
        """
        Gets the dissimilarities from the `set_dissimilarities()` call. Will throw `NameError` if `set_dissimilarities()` has
        not been previously called.

        Returns:
            :obj:`torch.Tensor`: n-by-3-by-2 tensor of the forward, backward, and total dissimilarity with their respective point x.

        Example::

            n = torch.normal(mean=0., std=1., size=(20,2)) # 2-dimensional data
            r = RulsifTimeSeriesAnomalyDetection(n, 10, 50, 0.569, 0.1, 0.01, 'cuda', 2048)
            r.set_dissimilarities() # sets the dissimilarities to RuLSIF dissimilarities
            r.get_dissimilarities() # gets the RuLSIF dissimilarities
        """
        return self.__dissimilarities

    def get_batch_dissimilarities(self) -> torch.Tensor:
        """
        Gets the dissimilarities from the `set_batch_dissimilarities()` call. Will throw `NameError` if `set_batch_dissimilarities()`
        has not been previously called.

        Returns:
            :obj:`torch.Tensor`: n-by-3-by-2 tensor of the forward, backward, and total dissimilarity with their respective point x.

        Example::

            n = torch.normal(mean=0., std=1., size=(20,2)) # 2-dimensional data
            r = RulsifTimeSeriesAnomalyDetection(n, 10, 50, 0.569, 0.1, 0.01, 'cuda', 2048)
            r.set_batch_dissimilarities() # sets the dissimilarities to RuLSIF dissimilarities
            r.get_batch_dissimilarities() # gets the RuLSIF dissimilarities
        """
        return self.__batch_dissimilarities

    def set_device(self, device: str) -> None:
        """
        Sets the device to store dissimilarities and perform computation.
        """
        self.__device = device

    def get_device(self) -> str:
        """
        Gets the device to perform dissimilarities computation and store the dissimilarities. Note that the data is stored on the
        host.

        Returns:
            :obj:`torch.device`: torch device object.
        """
        return self.__device

    def set_data(self, data: torch.Tensor) -> None:
        """
        Sets the data to perform dissimilarities computaiton. This will set the data
        """
        self.__data = data.to(device=self.__device)
        self.__dissimilarities_size = (
            self.__data.shape[0] - self.__sample_width - (2 * self.__retro_width) + 2
        )

    def get_data(self) -> torch.Tensor:
        """
        Gets the raw data as a torch tensor.

        Returns:
            :obj:`torch.Tensor`: torch.Tensor object of the multi-dimensional data.
        """
        return self.__data

    def set_sample_width(self, sample_width: int | torch.Tensor) -> None:
        """
        Sets the sample width before computation.
        """
        self.__sample_width = sample_width

    def get_sample_width(self) -> int | torch.Tensor:
        """
        Get the sample width. This is also known as the `k` value.

        Returns:
            :obj:`int`: sample width (k).
        """
        return self.__sample_width

    def set_retro_width(self, retro_width: int | torch.Tensor) -> None:
        """
        Sets the retro width before the computation.
        """
        self.__retro_width = retro_width

    def get_retro_width(self) -> int | torch.Tensor:
        """
        Get the retro width. This is also known as the `n` value.

        Returns:
            :obj:`int`: retro width (n).
        """
        return self.__retro_width

    def set_sigma(self, sigma: float | torch.Tensor) -> None:
        """
        Sets the sigma value before the computation.
        """
        self.__sigma = sigma

    def get_sigma(self) -> float | torch.Tensor:
        """
        Get the sigma set for the kernel.

        Returns:
            :obj:`float`: sigma.
        """
        return self.__sigma

    def set_alpha(self, alpha: int | torch.Tensor) -> None:
        """
        Sets the alpha value before the computation.
        """
        self.__alpha = alpha

    def get_alpha(self) -> float | torch.Tensor:
        """
        Get alpha set for the ratio of comparisons.

        Returns:
            :obj:`float`: alpha.
        """
        return self.__alpha

    def set_lambda(self, lambda_: int | torch.Tensor) -> None:
        """
        Sets the lambda value before the computation.
        """
        self.__lambda = lambda_

    def get_lambda(self) -> float | torch.Tensor:
        """
        Get lambda set for regularization.

        Returns:
            :obj:`float`: lambda.
        """
        return self.__lambda

    def set_batch_size(self, batch_size: int | torch.Tensor) -> None:
        """
        Sets the batch size before the computation.
        """
        self.__batch_size = batch_size

    def get_batch_size(self) -> int | torch.Tensor:
        """
        Get the batch size used for `get_batch_dissimilarities()`. This can be used for benchmarking.

        Returns:
            :obj:`int`: batch size.
        """
        return self.__batch_size

    def get_dissimilarities_size(self) -> int | torch.Tensor:
        """
        Get the dissimilarity size. Note that this is less than the size of the input data size.

        Returns:
            :obj:`int`: `N - k - 2n + 2`, where `N` is the data size, `k` is the sample width, and `n` is the retro width.
        """
        return self.__dissimilarities_size

    def get_offset(self) -> int | torch.Tensor:
        """
        Get the starting offset in which the dissimilarities begin recording.

        Returns:
            :obj:`int`: `k//2 + n` where `k` is the sample width and `n` is the retro width.
        """
        return self.__retro_width + self.__sample_width // 2

    @staticmethod
    def generate_torch_k_gaussian_kernel(
        sigma: float | torch.Tensor, dim=1
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Generates a 1-dimensional gaussian kernel from the sigma input. Takes the vector norm in the 1st
        dimension.

        Args:
            sigma (:obj:`float`): sigma to set the Gaussian kernel method.

        Returns:
            :obj:`Callable[[torch.Tensor], torch.Tensor]`: callable lambda function.
        """
        return lambda x: torch.exp(
            -(
                torch.square(torch.linalg.vector_norm(x, dim=dim))
                / (2.0 * sigma * sigma)
            )
        )

    @staticmethod
    def torch_vectorized_matrix_subtract(
        m_1: torch.Tensor, m_2: torch.Tensor
    ) -> torch.Tensor:
        return m_1[:, None, :] - m_2[None, :, :]

    @staticmethod
    def torch_vectorized_matrix_matrix_subtract(
        m_1: torch.Tensor, m_2: torch.Tensor
    ) -> torch.Tensor:
        return m_1[:, :, None, :] - m_2[:, None, :, :]

    @staticmethod
    def torch_kernel_density_ratio_model(
        y_1: torch.Tensor,
        y_2: torch.Tensor,
        thetas: torch.Tensor,
        kernel: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        diff = RulsifTimeSeriesAnomalyDetection.torch_vectorized_matrix_subtract(
            y_1, y_2
        )
        K = kernel(diff)
        return K @ thetas

    @staticmethod
    def torch_kernel_density_ratio_model2(
        y_1: torch.Tensor,
        y_2: torch.Tensor,
        thetas: torch.Tensor,
        kernel: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        diff = RulsifTimeSeriesAnomalyDetection.torch_vectorized_matrix_matrix_subtract(
            y_1, y_2
        )
        K = kernel(diff)
        return RulsifTimeSeriesAnomalyDetection.broadcast_matrix_vector_mult(K, thetas)

    @staticmethod
    def torch_H(
        y_1: torch.Tensor,
        y_2: torch.Tensor,
        alpha: float | torch.Tensor,
        kernel: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        n = y_1.shape[0]
        diff = RulsifTimeSeriesAnomalyDetection.torch_vectorized_matrix_subtract(
            y_1, y_1
        )
        K = kernel(diff)
        H1 = K.T @ K
        diff = RulsifTimeSeriesAnomalyDetection.torch_vectorized_matrix_subtract(
            y_2, y_1
        )
        K = kernel(diff)
        H2 = K.T @ K
        return ((alpha / n) * (H1)) + (((1.0 - alpha) / n) * (H2))

    @staticmethod
    def torch_H2(
        y_1: torch.Tensor,
        y_2: torch.Tensor,
        alpha: float | torch.Tensor,
        kernel: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        n = y_1.shape[1]
        diff = RulsifTimeSeriesAnomalyDetection.torch_vectorized_matrix_matrix_subtract(
            y_1, y_1
        )
        K = kernel(diff)
        H1 = torch.transpose(K, 1, 2) @ K
        diff = RulsifTimeSeriesAnomalyDetection.torch_vectorized_matrix_matrix_subtract(
            y_2, y_1
        )
        K = kernel(diff)
        H2 = torch.transpose(K, 1, 2) @ K
        return ((alpha / n) * (H1)) + (((1.0 - alpha) / n) * (H2))

    @staticmethod
    def torch_h(
        y_1: torch.Tensor,
        y_2: torch.Tensor,
        kernel: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        diff = RulsifTimeSeriesAnomalyDetection.torch_vectorized_matrix_subtract(
            y_1, y_2
        )
        K = kernel(diff)
        return K.sum(dim=0) / y_1.shape[0]

    @staticmethod
    def torch_h2(
        y_1: torch.Tensor,
        y_2: torch.Tensor,
        kernel: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        diff = RulsifTimeSeriesAnomalyDetection.torch_vectorized_matrix_matrix_subtract(
            y_1, y_2
        )
        K = kernel(diff)
        return K.sum(dim=1) / y_1.shape[1]

    @staticmethod
    def torch_get_thetas(
        y_1: torch.Tensor,
        y_2: torch.Tensor,
        kernel: Callable[[torch.Tensor], torch.Tensor],
        alpha: float | torch.Tensor,
        _lambda: float | torch.Tensor,
    ) -> torch.Tensor:
        n = y_1.shape[0]
        H = RulsifTimeSeriesAnomalyDetection.torch_H(y_1, y_2, alpha, kernel)
        h = RulsifTimeSeriesAnomalyDetection.torch_h(y_1, y_1, kernel)
        return torch.linalg.solve(
            H + _lambda * torch.eye(n, dtype=torch.int32, device=h.device), h
        )

    @staticmethod
    def torch_get_thetas2(
        y_1: torch.Tensor,
        y_2: torch.Tensor,
        kernel: Callable[[torch.Tensor], torch.Tensor],
        alpha: float | torch.Tensor,
        _lambda: float | torch.Tensor,
    ) -> torch.Tensor:
        n = y_1.shape[1]
        H = RulsifTimeSeriesAnomalyDetection.torch_H2(y_1, y_2, alpha, kernel)
        h = RulsifTimeSeriesAnomalyDetection.torch_h2(y_1, y_1, kernel)
        return torch.linalg.solve(
            H + _lambda * torch.eye(n, dtype=torch.int32, device=h.device), h
        )

    @staticmethod
    def torch_PE(
        y_1: torch.Tensor,
        y_2: torch.Tensor,
        thetas: torch.Tensor,
        alpha: float | torch.Tensor,
        kernel: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        n = y_1.shape[0]
        g1 = RulsifTimeSeriesAnomalyDetection.torch_kernel_density_ratio_model(
            y_1, y_1, thetas, kernel
        )
        g2 = RulsifTimeSeriesAnomalyDetection.torch_kernel_density_ratio_model(
            y_2, y_1, thetas, kernel
        )
        return (
            -((alpha / (2.0 * n)) * torch.sum(torch.square(g1)))
            - (((1.0 - alpha) / (2.0 * n)) * torch.sum(torch.square(g2)))
            + ((1.0 / n) * torch.sum(g1))
            - (0.5)
        )

    @staticmethod
    def torch_PE2(
        y_1: torch.Tensor,
        y_2: torch.Tensor,
        thetas: torch.Tensor,
        alpha: float | torch.Tensor,
        kernel: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        n = y_1.shape[1]
        g1 = RulsifTimeSeriesAnomalyDetection.torch_kernel_density_ratio_model2(
            y_1, y_1, thetas, kernel
        )
        g2 = RulsifTimeSeriesAnomalyDetection.torch_kernel_density_ratio_model2(
            y_2, y_1, thetas, kernel
        )
        return (
            -((alpha / 2.0) * torch.mean(torch.square(g1), dim=1))
            - (((1 - alpha) / 2.0) * torch.mean(torch.square(g2), dim=1))
            + (torch.mean(g1, dim=1))
            - (0.5)
        )

    @staticmethod
    def J_of_thetas(
        y_1: torch.Tensor,
        y_2: torch.Tensor,
        thetas: torch.Tensor,
        alpha: float | torch.Tensor,
        kernel: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        g1 = RulsifTimeSeriesAnomalyDetection.torch_kernel_density_ratio_model(
            y_1, y_1, thetas, kernel
        )
        g2 = RulsifTimeSeriesAnomalyDetection.torch_kernel_density_ratio_model(
            y_2, y_1, thetas, kernel
        )
        return (
            ((alpha / 2.0) * torch.mean(g1 * g1))
            + (((1.0 - alpha) / 2.0) * torch.mean(g2 * g2))
            - torch.mean(g1)
        )

    @staticmethod
    def J_of_thetas2(
        y_1: torch.Tensor,
        y_2: torch.Tensor,
        thetas: torch.Tensor,
        alpha: float | torch.Tensor,
        kernel: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        g1 = RulsifTimeSeriesAnomalyDetection.torch_kernel_density_ratio_model2(
            y_1, y_1, thetas, kernel
        )
        g2 = RulsifTimeSeriesAnomalyDetection.torch_kernel_density_ratio_model2(
            y_2, y_1, thetas, kernel
        )
        return (
            ((alpha / 2.0) * torch.mean(g1 * g1, dim=1))
            + (((1 - alpha) / 2.0) * torch.mean(g2 * g2, dim=1))
            - torch.mean(g1, dim=1)
        )

    @staticmethod
    def calculate_dissimilarity(
        y_1: torch.Tensor,
        y_2: torch.Tensor,
        alpha: float | torch.Tensor,
        _lambda: float | torch.Tensor,
        kernel: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        thetas = RulsifTimeSeriesAnomalyDetection.torch_get_thetas(
            y_1, y_2, kernel, alpha, _lambda
        )
        divergence = RulsifTimeSeriesAnomalyDetection.torch_PE(
            y_1, y_2, thetas, alpha, kernel
        )
        return divergence

    @staticmethod
    def calculate_dissimilarity2(
        y_1: torch.Tensor,
        y_2: torch.Tensor,
        alpha: float | torch.Tensor,
        _lambda: float | torch.Tensor,
        kernel: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        thetas = RulsifTimeSeriesAnomalyDetection.torch_get_thetas2(
            y_1, y_2, kernel, alpha, _lambda
        )
        divergence = RulsifTimeSeriesAnomalyDetection.torch_PE2(
            y_1, y_2, thetas, alpha, kernel
        )
        return divergence

    @staticmethod
    def bidirectional_dissimilarity(
        y_1: torch.Tensor,
        y_2: torch.Tensor,
        alpha: float | torch.Tensor,
        _lambda: float | torch.Tensor,
        kernel: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        forward = RulsifTimeSeriesAnomalyDetection.calculate_dissimilarity(
            y_1, y_2, alpha, _lambda, kernel
        )
        backwards = RulsifTimeSeriesAnomalyDetection.calculate_dissimilarity(
            y_2, y_1, alpha, _lambda, kernel
        )
        return torch.vstack([forward, backwards, forward + backwards])

    @staticmethod
    def bidirectional_dissimilarity2(
        y_1: torch.Tensor,
        y_2: torch.Tensor,
        alpha: float | torch.Tensor,
        _lambda: float | torch.Tensor,
        kernel: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        forward = RulsifTimeSeriesAnomalyDetection.calculate_dissimilarity2(
            y_1, y_2, alpha, _lambda, kernel
        )
        backwards = RulsifTimeSeriesAnomalyDetection.calculate_dissimilarity2(
            y_2, y_1, alpha, _lambda, kernel
        )
        return torch.vstack((forward, backwards, forward + backwards))

    @staticmethod
    def fast_bidirectional_dissimilarity2(
        y_1: torch.Tensor,
        y_2: torch.Tensor,
        alpha: float | torch.Tensor,
        _lambda: float | torch.Tensor,
        kernel: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        n11 = torch.tensor(y_1.shape[1], dtype=torch.int32, device=y_1.device)
        one = torch.tensor(1.0, dtype=torch.float32, device=y_1.device)
        two = torch.tensor(2.0, dtype=torch.float32, device=y_1.device)
        half = torch.tensor(0.5, dtype=torch.float32, device=y_1.device)
        f_1 = alpha / n11
        f_2 = (one - alpha) / n11
        l_i = _lambda * torch.eye(n11, dtype=torch.float32, device=y_1.device)
        f_3 = alpha / (two * n11)
        f_4 = (one - alpha) / (two * n11)
        f_5 = one / n11
        K11 = kernel(y_1[:, :, None] - y_1[:, None, :])
        K21 = kernel(y_2[:, :, None] - y_1[:, None, :])
        forward_thetas = torch.linalg.solve(
            (
                (f_1 * (torch.transpose(K11, 1, 2) @ K11))
                + ((f_2) * (torch.transpose(K21, 1, 2) @ K21))
                + l_i
            ),
            K11.sum(dim=1) / n11,
        )
        forward_g1 = torch.einsum("ijk,ik->ij", K11, forward_thetas)
        forward_PE = (
            -((f_3) * torch.sum(torch.square(forward_g1), dim=1))
            - (
                (f_4)
                * torch.sum(
                    torch.square(torch.einsum("ijk,ik->ij", K21, forward_thetas)), dim=1
                )
            )
            + (f_5 * torch.sum(forward_g1, dim=1))
            - (half)
        )
        K22 = kernel(y_2[:, :, None, :] - y_2[:, None, :, :])
        K12 = kernel(y_1[:, :, None, :] - y_2[:, None, :, :])
        backward_thetas = torch.linalg.solve(
            (
                (f_1 * (torch.transpose(K22, 1, 2) @ K22))
                + ((f_2) * (torch.transpose(K12, 1, 2) @ K12))
                + l_i
            ),
            K22.sum(dim=1) / n11,
        )
        backward_g1 = torch.einsum("ijk,ik->ij", K22, backward_thetas)
        backward_PE = (
            -((f_3) * torch.sum(torch.square(backward_g1), dim=1))
            - (
                (f_4)
                * torch.sum(
                    torch.square(torch.einsum("ijk,ik->ij", K12, backward_thetas)),
                    dim=1,
                )
            )
            + (f_5 * torch.sum(backward_g1, dim=1))
            - (half)
        )
        return torch.vstack((forward_PE, backward_PE, forward_PE + backward_PE))

    @staticmethod
    def broadcast_matrix_vector_mult(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ijk,ik->ij", A, B)

    @staticmethod
    def convert_data_to_windows(
        data: torch.Tensor, sample_width: int | torch.Tensor
    ) -> torch.Tensor:  # WORKING
        raw_subsequences = data.unfold(
            dimension=0, size=sample_width, step=1
        )  # should be N - k + 1
        shape = raw_subsequences.shape
        subsequences = (
            raw_subsequences.reshape(shape[0], shape[1] * shape[2])
            if len(data.shape) > 1
            else raw_subsequences
        )
        return subsequences

    @staticmethod
    def convert_data_to_windows_batches(
        data: torch.Tensor,
        sample_width: int | torch.Tensor,
        retro_width: int | torch.Tensor,
    ) -> torch.Tensor:  # WORKING
        data = data.view(data.shape[0], 1) if len(data.shape) < 2 else data
        retro_windows = data.unfold(dimension=0, size=retro_width, step=1)
        sample_windows = retro_windows.unfold(
            dimension=0, size=sample_width, step=1
        )  # should be N - n - k + 2
        retro_window_t = sample_windows.transpose(1, 2)
        shape = retro_window_t.shape
        batch_subsequences = retro_window_t.reshape(
            shape[0], shape[1], shape[2] * shape[3]
        )
        return batch_subsequences
