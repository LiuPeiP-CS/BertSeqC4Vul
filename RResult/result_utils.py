#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/7 上午10:59
# @Author  : PeiP Liu
# @FileName: result_utils.py
# @Software: PyCharm
import re
import json
from collections import defaultdict
import pandas as pd
import sys
sys.path.append('..')
from arguments import Args as args


def read_test(df_addr):
    dataset_dict = {}
    with open(df_addr, 'r', encoding='utf-8') as fr:
        content = fr.readlines()
        for line in content:
            if line.strip() is not '':
                dict = json.loads(line)
                dataset_dict[dict['cve-number']] = dict # cve-number, description, privilege-required, attack-vector, impact
    return dataset_dict


def read_json(df_addr):
    with open(df_addr, 'r', encoding='utf-8') as fr:
        dataset_dict = json.load(fr) # cve-number, description, privilege-required, attack-vector, impact
    return dataset_dict


if __name__ == '__main__':
    orig_test = args.test_addr
    test_dict = read_test(orig_test)

    bresult = args.bresult + args.setting + '.json'
    aresult = args.rresult + args.setting + '.xlsx'
    result_dict = read_json(bresult)

    for cve_id, cve_detail in test_dict.items():
        if cve_id not in result_dict:
            print(cve_id)
        else:
            # 描述中存在dos的，全部设置为dos;
            if 'dos' in cve_detail['description']:
                result_dict[cve_id]['impact'] = 'dos'
            # 存在unknown impact的，全部设置为unknown
            elif 'unknown impact' in cve_detail['description']:
                result_dict[cve_id]['impact'] = 'privileged-gained(rce)_unknown'

    num = []
    desp = []
    pr = []
    av = []
    il = []
    il1 = []
    il2 = []
    for cve_id, cve_detail in result_dict.items():
        num.append(cve_id)

        desp.append(cve_detail['description'])

        pr_str = cve_detail['privilege-required']
        if pr_str.lower() == 'admin/root':
            pr_str = 'admin/root'
        elif pr_str.lower() == 'nonprivileged':
            pr_str = 'Nonprivileged'
        elif pr_str.lower() == 'access':
            pr_str = 'access'
        elif pr_str.lower() == 'unknown':
            pr_str = 'unknown'
        pr.append(pr_str)

        av_str = cve_detail['attack-vector']
        if av_str.lower() == 'remote':
            av_str = 'remote'
        elif av_str.lower() == 'non-remote':
            av_str = 'Non-remote'
        av.append(av_str)

        if not args.O4C:
            il_str = cve_detail['impact']
            if il_str.lower() == 'other':
                il_str = 'other'
            elif il_str.lower() == 'access':
                il_str = 'access'
            elif il_str.lower() == 'dos':
                il_str = 'DoS'
            elif il_str.lower() == 'information-disclosure':
                il_str = 'Information-disclosure'
            elif 'privileged-gained(rce)' in il_str.lower():
                il_str = 'Privileged-Gained(RCE)'
            il.append(il_str)

            il1_str = ''
            if 'impact1' in cve_detail and cve_detail['impact1'] is not 'None': # 存在这个键，python2.X版本可以使用has_key()
                il1_str = cve_detail['impact1']
                if il1_str.lower() == 'privileged-gained(rce)_admin/root':
                    il1_str = 'admin/root'
                elif il1_str.lower() == 'privileged-gained(rce)_nonprivileged':
                    il1_str = 'Nonprivileged'
                elif il1_str.lower() == 'privileged-gained(rce)_unknown':
                    il1_str = 'unknown'
                elif il1_str.lower() == 'information-disclosure_other-target(credit)':
                    il1_str = 'other-target(credit)'
                elif il1_str.lower() == 'information-disclosure_local(credit)':
                    il1_str = 'local(credit)'
                elif il1_str.lower() == 'information-disclosure_other':
                    il1_str = 'other'
            elif cve_detail['impact'] == 'privileged-gained(rce)_unknown': # 后处理的补充方案
                il1_str = 'unknown'
            il1.append(il1_str)

            il2_str = ''
            if 'impact2' in cve_detail:
                il2_str = cve_detail['impact2']
                if il2_str.lower() == 'information-disclosure_local(credit)_nonprivileged':
                    il2_str = 'Nonprivileged'
                elif il2_str.lower() == 'information-disclosure_local(credit)_unknown':
                    il2_str = 'unknown'
                elif il2_str.lower() == 'information-disclosure_local(credit)_admin/root':
                    il2_str = 'admin/root'
                elif il2_str.lower() == 'information-disclosure_other-target(credit)_admin/root':
                    il2_str = 'admin/root'
                elif il2_str.lower() == 'information-disclosure_other-target(credit)_unknown':
                    il2_str = 'unknown'
                elif il2_str.lower() == 'information-disclosure_other-target(credit)_nonprivileged':
                    il2_str = 'Nonprivileged'
            il2.append(il2_str)

        else:
            il_str = cve_detail['impact']
            if il_str.lower() == 'other':
                il_str = 'other'
            elif il_str.lower() == 'access':
                il_str = 'access'
            elif il_str.lower() == 'dos':
                il_str = 'DoS'
            elif 'information-disclosure' in il_str.lower():
                il_str = 'Information-disclosure'
            elif 'privileged-gained(rce)' in il_str.lower():
                il_str = 'Privileged-Gained(RCE)'
            il.append(il_str)

            il1_str = cve_detail['impact']
            if 'privileged-gained(rce)_admin/root' in il1_str.lower():
                il1_str = 'admin/root'
            elif 'privileged-gained(rce)_nonprivileged' in il1_str.lower():
                il1_str = 'Nonprivileged'
            elif 'privileged-gained(rce)_unknown' in il1_str.lower():
                il1_str = 'unknown'
            elif 'information-disclosure_other-target(credit)' in il1_str.lower():
                il1_str = 'other-target(credit)'
            elif 'information-disclosure_local(credit)' in il1_str.lower():
                il1_str = 'local(credit)'
            elif 'information-disclosure_other' in il1_str.lower():
                il1_str = 'other'
            else:
                il1_str = ''
            il1.append(il1_str)

            il2_str = cve_detail['impact']
            if 'information-disclosure_local(credit)_nonprivileged' in il2_str.lower():
                il2_str = 'Nonprivileged'
            elif 'information-disclosure_local(credit)_unknown' in il2_str.lower():
                il2_str = 'unknown'
            elif 'information-disclosure_local(credit)_admin/root' in il2_str.lower():
                il2_str = 'admin/root'
            elif 'information-disclosure_other-target(credit)_admin/root' in il2_str.lower():
                il2_str = 'admin/root'
            elif 'information-disclosure_other-target(credit)_unknown' in il2_str.lower():
                il2_str = 'unknown'
            elif 'information-disclosure_other-target(credit)_nonprivileged' in il2_str.lower():
                il2_str = 'Nonprivileged'
            else:
                il2_str = ''
            il2.append(il2_str)

    wr_dict = {'CVE-Number': num,
               'Desription': desp,
               'Privilege-Required': pr,
               'Attack-Vector': av,
               'Impact-level1': il,
               'Impact-level2': il1,
               'Impact-level3': il2}

    fw = pd.DataFrame(wr_dict)
    fw.to_excel(aresult, index=False)