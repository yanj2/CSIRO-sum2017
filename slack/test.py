NUMDICT = {1:2,3:4}

def main():
    changeDict({2:1})

def changeDict(nums=NUMDICT):
    for key in nums:
        nums[key]

if __name__ == "__main__":
    main()
