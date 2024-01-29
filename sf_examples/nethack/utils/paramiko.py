import paramiko


def get_save_paths(
    dir,
    # SSH connection details
    host="athena.cyfronet.pl",
    username="plgbartekcupial",
    private_key_path="/home/bartek/.ssh/id_rsa",
):
    # Command to execute
    ls = f"ls {dir}"

    # Create SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        private_key = paramiko.RSAKey.from_private_key_file(private_key_path)

        # Connect to the SSH server
        ssh.connect(host, username=username, pkey=private_key)

        # Open an SFTP session
        sftp = ssh.open_sftp()

        # Execute the command
        stdin, stdout, stderr = ssh.exec_command(ls)

        # Read the output
        output = stdout.read().decode()

        paths = output.split("\n")
        # filter empty strings
        paths = list(filter(None, paths))

        # Close the SFTP session
        sftp.close()
        return paths
    finally:
        # Close the SSH connection
        ssh.close()


def get_checkpoint_paths(
    dir,
    # SSH connection details
    host="athena.cyfronet.pl",
    username="plgbartekcupial",
    private_key_path="/home/bartek/.ssh/id_rsa",
):
    # Command to execute
    find = f'find {dir} -type f -name "checkpoint_*" -printf "%h\n" | sort -u'

    # Create SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        private_key = paramiko.RSAKey.from_private_key_file(private_key_path)

        # Connect to the SSH server
        ssh.connect(host, username=username, pkey=private_key)

        # Open an SFTP session
        sftp = ssh.open_sftp()

        # Execute the command
        stdin, stdout, stderr = ssh.exec_command(find)

        # Read the output
        output = stdout.read().decode()

        paths = output.split("\n")
        # filter empty strings
        paths = list(filter(None, paths))

        # Close the SFTP session
        sftp.close()
        return paths
    finally:
        # Close the SSH connection
        ssh.close()
