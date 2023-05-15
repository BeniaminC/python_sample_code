"""
RuLSIF Time-Series Anomaly detection using torch
Author: Beniamin Condrea
Written 7/18/2022
"""

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
    Apply RuLSIF anomaly detection on multi-dimensional time series data. Supports batching.

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
        self.device = device
        self.data = data # data stays on host until computed

        self.sample_width = torch.tensor(
            sample_width, dtype=torch.int32, device=self.device
        )
        self.retro_width = torch.tensor(
            retro_width, dtype=torch.int32, device=self.device
        )
        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=self.device)
        self.alpha = torch.tensor(alpha, dtype=torch.float32, device=self.device)
        self.lambda_ = torch.tensor(_lambda, dtype=torch.float32, device=self.device)
        self.batch_size = torch.tensor(
            batch_size, dtype=torch.int32, device=self.device
        )
        # below is removed becuase the getter computes it, we can cache it if we really need to squeeze performance
        # self.dissimilarities_size = torch.tensor(
        #     self.data.shape[0] - self.sample_width - (2 * self.retro_width) + 2,
        #     dtype=torch.int32,
        #     device=self.device,
        # )
        self.__batch_dissimilarities = torch.tensor(())
    

    @property
    def batch_dissimilarities(self) -> torch.Tensor:
        """
        Gets the dissimilarities from the `compute_batch_dissimilarities()` call.
        Returns:
            :obj:`torch.Tensor`: n-by-4 tensor of the x-point, forward, backward, and total dissimilarity with their respective point x.

        Example::

            n = torch.normal(mean=0., std=1., size=(20,2)) # 2-dimensional data
            r = RulsifTimeSeriesAnomalyDetection(n, 10, 50, 0.569, 0.1, 0.01, 'cuda', 2048)
            r.compute_batch_dissimilarities() # sets the dissimilarities to RuLSIF dissimilarities
            r.batch_dissimilarities # gets the RuLSIF dissimilarities
        """
        return self.__batch_dissimilarities

    def compute_batch_dissimilarities(self) -> None:
        """
        Sets the dissimilarities with batching. May more effectively use the GPU. Will perform faster than `dissimilarities()`.
        Larger batch size will generally improve GPU utilization. Will throw `RuntimeError: CUDA out of memory.` if batch size is too large.

        Example::

            n = torch.normal(mean=0., std=1., size=(20,2)) # 2-dimensional data
            r = RulsifTimeSeriesAnomalyDetection(n, 10, 50, 0.569, 0.1, 0.01, 'cuda', 2048)
            r.compute_batch_dissimilarities() # sets the dissimilarities to RuLSIF dissimilarities
        """
        try:
            assert (
                self.dissimilarities_size > 0
            ), f"""Cannot run Rulsif algorithm under these conditions.
                Dissimilarity size: {self.dissimilarities_size}.
                Input more data, or decrease the sample width and/or retro width"""
        except AssertionError as err:
            raise err
        self.__batch_dissimilarities = torch.zeros((self.dissimilarities_size, 4), device=self.device)
        sigma_gaussian_kernel_d3 = RulsifTimeSeriesAnomalyDetection.generate_torch_k_gaussian_kernel(
            self.sigma, dim=3
        )
        batch_subsequences = (
            RulsifTimeSeriesAnomalyDetection.convert_data_to_windows_batches(
                self.data, self.sample_width, self.retro_width
            )
        )
        offset = self.offset
        assert (
            self.batch_size > 0
        ), f"Batch size must be at least 1. Current batch size: {self.batch_size}"
        try:
            start_idx = torch.tensor(0, dtype=torch.int32, device=self.device)
            end_idx = torch.tensor(self.batch_size, device=self.device)
            y_1_batch_subsequences = batch_subsequences[: self.dissimilarities_size]
            y_2_batch_subsequences = batch_subsequences[self.retro_width :]
            assert y_1_batch_subsequences.shape[0] == y_2_batch_subsequences.shape[0]
            while start_idx < self.dissimilarities_size:
                y_1 = y_1_batch_subsequences[start_idx:end_idx]
                y_2 = y_2_batch_subsequences[start_idx:end_idx]
                diss_calc = RulsifTimeSeriesAnomalyDetection.fast_batch_bidirectional_dissimilarity(
                    y_1, y_2, self.alpha, self.lambda_, sigma_gaussian_kernel_d3
                )
                idx_offset = start_idx + offset
                point_x = torch.arange(
                    idx_offset,
                    y_1.shape[0] + idx_offset,
                    1,
                    dtype=torch.float32,
                    device=self.device,
                )
                diss_data = torch.column_stack(
                    (point_x, diss_calc[0], diss_calc[1], diss_calc[2])
                )
                self.batch_dissimilarities[start_idx:end_idx] = diss_data
                start_idx += self.batch_size
                end_idx += self.batch_size
        except RuntimeError as err:
            torch.cuda.ipc_collect()
            raise err

    @property
    def device(self) -> str:
        """
        Gets the device to perform dissimilarities computation and store the dissimilarities. Note that the data is stored on the
        host.

        Returns:
            :obj:`torch.device`: torch device object.
        """
        return self.__device

    # should use descriptors for setters/getters (DRY)
    @device.setter
    def device(self, device: str) -> None:
        """
        Sets the device to store dissimilarities and perform computation.
        """
        self.__device = device
        # if we are changing devices, don't forget the data
        try:
            if hasattr(self, '__data') and self.__device not in self.__data.device.type:
                self.data = self.device.to(self.device)
        except ValueError as err:
            raise err

    @property
    def data(self) -> torch.Tensor:
        """
        Gets the raw data as a torch tensor.

        Returns:
            :obj:`torch.Tensor`: torch.Tensor object of the multi-dimensional data.
        """
        return self.__data

    @data.setter
    def data(self, data: torch.Tensor) -> None:
        """
        Sets the data to perform dissimilarities computaiton. This will set the data
        """
        try:
            self.__data = data.to(device=self.device)
        except ValueError as err:
            raise err

    @property
    def sample_width(self) -> int | torch.Tensor:
        """
        Get the sample width. This is also known as the `k` value.

        Returns:
            :obj:`int`: sample width (k).
        """
        return self.__sample_width

    @sample_width.setter
    def sample_width(self, sample_width: int | torch.Tensor) -> None:
        """
        Sets the sample width before computation.
        """
        try:
            self.__sample_width = int(sample_width)
            if self.__sample_width < 1:
                raise ValueError
        except ValueError as err:
            raise err
        

    @property
    def retro_width(self) -> int | torch.Tensor:
        """
        Get the retro width. This is also known as the `n` value.

        Returns:
            :obj:`int`: retro width (n).
        """
        return self.__retro_width

    @retro_width.setter
    def retro_width(self, retro_width: int | torch.Tensor) -> None:
        """
        Sets the retro width before the computation.
        """
        try:
            self.__retro_width = int(retro_width)
            if retro_width < 1:
                raise ValueError
        except ValueError as err:
            raise err

    @property
    def sigma(self) -> float | torch.Tensor:
        """
        Get the sigma set for the kernel.

        Returns:
            :obj:`float`: sigma.
        """
        return self.__sigma

    @sigma.setter
    def sigma(self, sigma: float | torch.Tensor) -> None:
        """
        Sets the sigma value before the computation.
        """
        try:
            if sigma < 0:
                raise ValueError
            self.__sigma = torch.tensor(sigma, dtype=torch.float32, device=self.device)
        except ValueError as err:
            raise err

    @property
    def alpha(self) -> float | torch.Tensor:
        """
        Get alpha set for the ratio of comparisons.

        Returns:
            :obj:`float`: alpha.
        """
        return self.__alpha

    @alpha.setter
    def alpha(self, alpha: float | torch.Tensor) -> None:
        """
        Sets the alpha value before the computation.
        """
        try:
            if alpha < 0:
                raise ValueError
            self.__alpha = torch.tensor(alpha, dtype=torch.float32, device=self.device)
        except ValueError as err:
            raise err

    @property
    def lambda_(self) -> float | torch.Tensor:
        """
        Get lambda set for regularization.

        Returns:
            :obj:`float`: lambda.
        """
        return self.__lambda

    @lambda_.setter
    def lambda_(self, lambda_: float | torch.Tensor) -> None:
        """
        Sets the lambda value before the computation.
        """
        try:
            self.__lambda = torch.tensor(lambda_, dtype=torch.float32, device=self.device)
            if lambda_ < 0:
                raise ValueError
        except ValueError as err:
            raise err

    @property
    def batch_size(self) -> int | torch.Tensor:
        """
        Get the batch size used for `batch_dissimilarities()`. This can be used for benchmarking.

        Returns:
            :obj:`int`: batch size.
        """
        return self.__batch_size
    
    @batch_size.setter
    def batch_size(self, batch_size: int | torch.Tensor) -> None:
        """
        Sets the batch size before the computation.
        """
        try:
            self.__batch_size = torch.tensor(batch_size, dtype=torch.int32, device=self.device)
            if batch_size < 1:
                raise ValueError
        except ValueError as err:
            raise err

    @property
    def offset(self) -> int | torch.Tensor:
        """
        Get the starting offset in which the dissimilarities begin recording.

        Returns:
            :obj:`int`: `k//2 + n` where `k` is the sample width and `n` is the retro width.
        """
        return self.retro_width + self.sample_width // 2
    
    @offset.setter
    def offset(self, offset: int | torch.Tensor) -> None:
        raise AttributeError('Cannot set the offset. Offset is computed from the retro_width and sample_width.')
    
    @property
    def dissimilarities_size(self) -> int | torch.Tensor:
        """
        Get the dissimilarity size. Note that this is less than the size of the input data size.

        Returns:
            :obj:`int`: `N - k - 2n + 2`, where `N` is the data size, `k` is the sample width, and `n` is the retro width.
        """
        return self.data.shape[0] - self.sample_width - (2 * self.retro_width) + 2
    
    @dissimilarities_size.setter
    def dissimilarities_size(self, size: int | torch.Tensor) -> None:
        raise AttributeError("Cannot set dissimilarity size. Dissimilarity size is computed from data, samplewidth and retro_width")

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

    # same as batch_bidirectional_dissimilarity, but without uneccesary duplicate computations and function calls
    @staticmethod
    def fast_batch_bidirectional_dissimilarity(
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

