from raagent import RLRA


if __name__ == "__main__":
    agent = RLRA()
    agent.initial_base()
    agent.setup_vs()
    print("You RA Agent is ready to help!")
    title = input("Enter what you are looking for : ")
    type = input("What type of information do you need? Options: Type 1 for code, 2 for method, 3 for Results and 4 for Empirical Performance: ")
    result = agent.ask(title, int(type))
    print(result)
    