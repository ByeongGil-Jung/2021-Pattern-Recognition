"""
2010322
01_Bayes_Rule
"""


def main():
    # Prior #
    # 상자가 파란색일 확률
    prob_blue_box = 6 / 10
    # 상자가 빨간색일 확률
    prob_red_box = 4 / 10  # 1 - prob_blue_Box

    # Likelihood #
    prob_apple_blue_box = 3 / 4
    prob_orange_blue_box = 1 / 4  # 1 - prob_apple_blue_box

    prob_apple_red_box = 2 / 8
    prob_orange_red_box = 6 / 8  # 1 - prob_apple_red_box

    # Prior (marginal prob) #
    prob_apple = (prob_blue_box * prob_apple_blue_box) + (prob_red_box * prob_apple_red_box)
    print(f"꺼냈는데 사과일 확률 : p(a) = {prob_apple}")
    prob_orange = 1 - prob_apple
    print(f"꺼냈는데 오렌지일 확률 : p(o) = {prob_orange}")

    # Posterior #
    prob_red_box_orange = (prob_orange_red_box * prob_red_box) / prob_orange
    print(f"p(r|o) = {prob_red_box_orange}")
    prob_blue_box_orange = 1 - prob_red_box_orange
    print(f"p(b|o) = {prob_blue_box_orange}")

    prob_red_box_apple = (prob_apple_red_box * prob_red_box) / prob_apple
    print(f"p(r|a) = {prob_red_box_apple}")
    prob_blue_box_apple = (prob_apple_blue_box * prob_blue_box) / prob_apple
    print(f"p(b|a) = {prob_blue_box_apple}")

    """
    꺼냈는데 사과일 확률 : p(a) = 0.5499999999999999
    꺼냈는데 오렌지일 확률 : p(o) = 0.45000000000000007
    p(r|o) = 0.6666666666666666
    p(b|o) = 0.33333333333333337
    p(r|a) = 0.18181818181818185
    p(b|a) = 0.8181818181818182
    """


if __name__ == "__main__":
    main()
