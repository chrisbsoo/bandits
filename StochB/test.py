import env

MAB = env.bandit_model("dnorm")
MAB.add("2 3", "0.1 0.2")
MAB.play(0)