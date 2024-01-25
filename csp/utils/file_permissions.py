import grp
import os
import pwd
import random
import time

from csp.impl.struct import Struct


class RWXPermissions:
    READ = 0x4
    WRITE = 0x2
    EXECUTE = 0x1


class FilePermissions(Struct):
    owner_user: str
    owner_group: str
    user_permissions: int
    group_permissions: int
    others_permissions: int

    def get_folder_permissions(self):
        res = self.copy()
        if getattr(res, "user_permissions"):
            res.user_permissions |= RWXPermissions.EXECUTE
        if getattr(res, "group_permissions"):
            res.group_permissions |= RWXPermissions.EXECUTE
        if getattr(res, "others_permissions"):
            res.others_permissions |= RWXPermissions.EXECUTE
        return res

    @classmethod
    def _postprocess_dict_to_python(cls, d):
        for k in ["user_permissions", "group_permissions", "others_permission"]:
            if k in d:
                d[k] = hex(d[k])
        return d

    @classmethod
    def _preprocess_dict_from_python(cls, d):
        for k in ["user_permissions", "group_permissions", "others_permission"]:
            if k in d:
                d[k] = int(d[k], 16)
        return d


def _get_uid(user_name):
    # There are some random sporadic errors in getpwnam, so we need to retry
    for i in range(5):
        try:
            return pwd.getpwnam(user_name).pw_uid
        except KeyError:
            time.sleep(random.random())
    return pwd.getpwnam(user_name).pw_uid


def _get_gid(group_name):
    # There are some random sporadic errors in getgrnam, so we need to retry
    for i in range(5):
        try:
            return grp.getgrnam(group_name).gr_gid
        except KeyError:
            time.sleep(random.random())
    return grp.getgrnam(group_name).gr_gid


def apply_file_permissions(file_path, file_permissions: FilePermissions):
    file_stat = os.stat(file_path)

    old_perm = file_stat.st_mode & 0xFFF
    new_perm = old_perm

    if hasattr(file_permissions, "user_permissions"):
        new_perm = new_perm & 0x77 | (file_permissions.user_permissions << 6)
    if hasattr(file_permissions, "group_permissions"):
        new_perm = new_perm & 0x1C7 | (file_permissions.group_permissions << 3)
    if hasattr(file_permissions, "others_permissions"):
        new_perm = new_perm & 0x1F8 | (file_permissions.others_permissions)
    if old_perm != new_perm:
        try:
            os.chmod(file_path, new_perm)
        except PermissionError:
            pass

    if hasattr(file_permissions, "owner_user") or hasattr(file_permissions, "owner_group"):
        uid, gid = file_stat.st_uid, file_stat.st_gid
        if hasattr(file_permissions, "owner_user"):
            uid = _get_uid(file_permissions.owner_user)
        if hasattr(file_permissions, "owner_group"):
            gid = _get_gid(file_permissions.owner_group)
        try:
            os.chown(file_path, uid=uid, gid=gid)
        except PermissionError:
            pass


def ensure_file_exists_with_permissions(file_path: str, file_permissions: FilePermissions):
    if not os.path.exists(file_path):
        with open(file_path, "a+"):
            pass

    apply_file_permissions(file_path=file_path, file_permissions=file_permissions)


def create_folder_with_permissions(folder_path: str, file_permissions: FilePermissions):
    """
    :param folder_path: The path of the folder to create
    :param file_permissions: The permissions of the files in the folder (the folder permissions are derived from it)
    """
    if os.path.exists(folder_path):
        return

    cur_path = os.path.abspath(os.sep)
    path_parts = os.path.normpath(os.path.abspath(folder_path)).split(os.sep)
    folder_permissions = file_permissions.get_folder_permissions()
    for sub_folder in path_parts:
        if not sub_folder:
            continue
        cur_path = os.path.join(cur_path, sub_folder)
        if os.path.exists(cur_path):
            continue
        try:
            os.mkdir(cur_path)
        except FileExistsError:
            pass

        apply_file_permissions(cur_path, folder_permissions)
