class Solution(object):
    def main(self, n):
        if n <= 6:
            return n

        i2, i3, i5 = 0, 0, 0
        res = [1]
        cur_num = 1
        while cur_num < n:
            min_value = min(res[i2] * 2, res[i3] * 3, res[i5] * 5)
            res.append(min_value)
            while res[i2] * 2 <= min_value:
                i2 += 1
            while res[i3] * 3 <= min_value:
                i3 += 1
            while res[i5] * 5 <= min_value:
                i5 += 1

            cur_num += 1

        return res

m = Solution()
print(m.main(10))