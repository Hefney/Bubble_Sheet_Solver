import model_1
import model_2
import model_3
import model_4
def main():
    mode= input("Choose Model Mode")
    if mode == "1":
        model_1.main()
    elif mode == "2":
        model_2.main()
    elif mode == "3":
        model_3.main()
    else :
        model_4.main()
if __name__ == '__main__':
    main()