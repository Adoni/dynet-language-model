import subprocess


def main():
    command = [
            './train_rnnlm_mp',
            '--dynet-mem 4500',
            '--dynet-seed 1',
            '--train_file ./lm_train.data',
            '--dev_file ./lm_test.data',
            '--word_embedding_file /users2/xfsun/zhihu_data/83095/embedding.data',
            '--workers 10',
            '--iterations 10',
        ]
    command = ' '.join(command)
    print(command)
    subprocess.call(command, shell=True)


if __name__ == '__main__':
    main()
