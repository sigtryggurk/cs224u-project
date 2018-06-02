from colorama import Fore, Style, init
init()

def log_info(info):
    print(Fore.BLUE + "[INFO]: " + Style.RESET_ALL + info)

def log_warning(info):
    print(Fore.YELLOW + "[WARNING]: " + Style.RESET_ALL + info)

if __name__=="__main__":
    log_info("This is a test")
