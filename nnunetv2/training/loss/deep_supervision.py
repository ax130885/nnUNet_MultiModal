from torch import nn

# 深度監督損失函數包裝器，可以對一個list的模型輸出批量計算損失，最後加權求和。
class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        將損失函數包裝起來，使其可以應用於多個輸出。
        Forward 方法接受任意數量的輸入，每個輸入都應該是 tuple 或 list。
        每個 tuple/list 必須有相同的長度。
        損失會這樣計算：
        l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
        如果 weights 是 None，則所有權重 w 都為 1。
        """
        super(DeepSupervisionWrapper, self).__init__()
        # 檢查至少有一個權重不為 0，否則報錯
        assert any([x != 0 for x in weight_factors]), "At least one weight factor should be != 0.0"
        self.weight_factors = tuple(weight_factors)  # 權重因子轉為 tuple 方便後續使用
        self.loss = loss  # 儲存傳入的損失函數

    def forward(self, *args):
        # 檢查所有輸入都是 tuple 或 list 型態
        assert all([isinstance(i, (tuple, list)) for i in args]), \
            f"all args must be either tuple or list, got {[type(i) for i in args]}"
        # 這裡可以檢查長度是否一致，但為了效能考量不做過多檢查

        # 如果沒有指定權重，則預設每個權重為 1
        if self.weight_factors is None:
            weights = (1, ) * len(args[0])
        else:
            weights = self.weight_factors

        # 計算每一個輸出分支的損失，並乘上對應權重後加總
        # zip(*args) 會將每個分支的對應輸出組成一個 tuple
        # 例如：args = ([a1, a2], [b1, b2]) -> zip(*args) = [(a1, b1), (a2, b2)]
        # enumerate 用於取得索引 i 及對應的 inputs
        # 只有當權重不為 0 時才計算該分支的損失
        return sum([weights[i] * self.loss(*inputs) for i, inputs in enumerate(zip(*args)) if weights[i] != 0.0])
