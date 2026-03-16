from gc import collect
from collector import collect_data

def main():
    collect_data(10000)
    print("Done data collection")

if __name__ == "__main__":
    main()