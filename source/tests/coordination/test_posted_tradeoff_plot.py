"""Synthetic plot data only; no saved or future test panel is read."""
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
import plot_posted_tradeoff as tradeoff


def address(public,private):
    diagonal=public+private
    return -(diagonal*(diagonal+1)//2+private+1)


def write_fixture(directory,schemes,*,draws=4):
    directory.mkdir()
    rows=[]
    for index,scheme in enumerate(schemes):
        for weather in range(2):
            for private in range(draws):
                rows.append(dict(scheme=scheme,case_id=address(weather,100+4*weather+private),
                    actual_S_yuan=index*100+weather*20+(private-1.5)*10,
                    actual_platform_gain_yuan=index*60+weather*10+(private-1.5)*5))
    pd.DataFrame(rows).to_csv(directory/'test_outcomes.csv',index=False,encoding='utf-8-sig')
    (directory/'test_report.json').write_text(json.dumps(dict(model_metadata=dict(history=True),
        extra_models={'no_history_ppo':dict(history=False)})),encoding='utf-8')
    return rows


def test_private_draws_are_averaged_before_weather_statistics(tmp_path):
    rows=write_fixture(tmp_path/'cohort',['exit'])
    cohort=tradeoff.load_tradeoff_cohort(tmp_path/'cohort','Synthetic',expected_weather=2,required_schemes=())
    point=cohort['points'][0]
    assert point['weather_days']==2 and point['min_private_draws']==point['max_private_draws']==4
    assert point['S_mean_yuan']==pytest.approx(10)
    assert point['profit_mean_yuan']==pytest.approx(5)
    expected=tradeoff.interval(np.array([0.,20.]))
    assert point['S_ci95_half_yuan']==pytest.approx(expected['half'])
    assert point['S_ci95_half_yuan']!=pytest.approx(tradeoff.interval([row['actual_S_yuan'] for row in rows])['half'])


def test_unbalanced_synthetic_draws_still_weight_weather_equally(tmp_path):
    directory=tmp_path/'cohort';directory.mkdir()
    rows=[dict(scheme='exit',case_id=address(0,100),actual_S_yuan=0.,actual_platform_gain_yuan=0.)]
    rows.extend(dict(scheme='exit',case_id=address(1,private),actual_S_yuan=10.,actual_platform_gain_yuan=4.)
                for private in (101,102,103))
    pd.DataFrame(rows).to_csv(directory/'test_outcomes.csv',index=False)
    cohort=tradeoff.load_tradeoff_cohort(directory,'Synthetic',expected_weather=2,expected_draws=None,required_schemes=())
    assert cohort['points'][0]['S_mean_yuan']==5.
    with pytest.raises(ValueError,match='4 private draws'):
        tradeoff.load_tradeoff_cohort(directory,'Synthetic',expected_weather=2,required_schemes=())


def test_nondominated_marks_observed_means_and_preserves_duplicate_controls():
    points=[dict(S_mean_yuan=x,profit_mean_yuan=y) for x,y in ((1,1),(2,.5),(.5,2),(3,3),(3,3))]
    assert tradeoff.nondominated_observed_means(points)==[False,False,False,True,True]


def test_all_controls_appear_in_separate_panel_legend_and_png_smoke(tmp_path):
    schemes=(*tradeoff.CORE_CONTROLS,'no_history_ppo')
    write_fixture(tmp_path/'initial',schemes)
    write_fixture(tmp_path/'joint',tradeoff.CORE_CONTROLS)
    cohorts=[tradeoff.load_tradeoff_cohort(tmp_path/'initial','Synthetic initial',expected_weather=2),
             tradeoff.load_tradeoff_cohort(tmp_path/'joint','Synthetic joint',expected_weather=2)]
    figure,points=tradeoff.create_tradeoff_figure(cohorts,dpi=90)
    try:
        assert len(figure.axes)==2 and len(points)==17
        assert sum(len(axis.texts) for axis in figure.axes)==17
        labels=' '.join(text.get_text() for text in figure.legends[0].get_texts())
        assert 'Fixed rebate matrix' in labels and 'online MAP' in labels
        assert 'independently trained' in labels and 'history zeroed' in labels
        assert len(figure.legends[0].get_texts())==10
        figure.savefig(tmp_path/'synthetic_tradeoff.png',dpi=90)
        assert (tmp_path/'synthetic_tradeoff.png').stat().st_size>10000
    finally:plt.close(figure)


def test_repeated_cases_and_missing_preset_controls_do_not_silently_disappear(tmp_path):
    rows=write_fixture(tmp_path/'cohort',['exit'])
    with pytest.raises(ValueError,match='Missing preset'):
        tradeoff.load_tradeoff_cohort(tmp_path/'cohort','Synthetic',expected_weather=2)
    pd.DataFrame([*rows,rows[0]]).to_csv(tmp_path/'cohort/test_outcomes.csv',index=False)
    with pytest.raises(ValueError,match='repeated'):
        tradeoff.load_tradeoff_cohort(tmp_path/'cohort','Synthetic',expected_weather=2,required_schemes=())
