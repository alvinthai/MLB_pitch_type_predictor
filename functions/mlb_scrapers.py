from selenium import webdriver
import numpy as np
import pandas as pd


def scrape_positions_2011():
    '''
    Scrapes positions of MLB players (2011 season) from https://www.baseball-reference.com
    '''
    fp = webdriver.FirefoxProfile()
    fp.set_preference("http.response.timeout", 5)
    fp.set_preference("dom.max_script_run_time", 5)

    driver = webdriver.Firefox(firefox_profile=fp)

    pos, players, urls = [], [], []

    driver.get('https://www.baseball-reference.com/leagues/NL/2011.shtml')
    xpath = "//table[@id[starts-with(., 'standings_')]]/tbody/tr/th[@data-stat='team_ID']"

    for page in driver.find_elements_by_xpath(xpath):
        urls.append(page.find_element_by_tag_name("a").get_attribute('href'))

    driver.get('https://www.baseball-reference.com/leagues/AL/2011.shtml')
    xpath = "//table[@id[starts-with(., 'standings_')]]/tbody/tr/th[@data-stat='team_ID']"

    for page in driver.find_elements_by_xpath(xpath):
        urls.append(page.find_element_by_tag_name("a").get_attribute('href'))

    if len(urls) != 30:
        raise AssertionError('Could not find HTML for all 30 MLB teams!')

    xpath1 = "//table[@id='team_batting']/tbody/tr/td[@data-stat='pos']"
    xpath2 = "//table[@id='team_batting']/tbody/tr/td[@data-stat='player']/a"

    for url in urls:
        driver.get(url)

        for x in driver.find_elements_by_xpath(xpath1):
            pos.append(x.text)

        for x in driver.find_elements_by_xpath(xpath2):
            players.append(x.text)

    driver.quit()

    table = np.array([players, pos]).T
    table = pd.DataFrame(table, columns=['player', 'position'])
    table = table.drop_duplicates('player')
    table = table.set_index('player')

    table.to_csv('../data/positions.csv')

    return table


def scrape_slg_2010():
    '''
    Scrapes SLG data of MLB players (2010 season) from https://www.baseball-reference.com
    SLG statistic info: http://www.wikiwand.com/en/Slugging_percentage
    '''
    driver = webdriver.Firefox()
    driver.get('https://www.baseball-reference.com/leagues/MLB/2010-standard-batting.shtml')

    names, slg, ab = [], [], []

    xpath = "//table[@id='players_standard_batting']/tbody/tr/td[@data-stat='player']/a"
    for x in driver.find_elements_by_xpath(xpath):
        names.append(x.text)

    xpath = "//table[@id='players_standard_batting']/tbody/tr/td[@data-stat='slugging_perc']"
    for x in driver.find_elements_by_xpath(xpath)[:-1]:
        slg.append(x.text)

    xpath = "//table[@id='players_standard_batting']/tbody/tr/td[@data-stat='AB']"
    for x in driver.find_elements_by_xpath(xpath)[:-1]:
        ab.append(x.text)

    driver.quit()

    table2 = np.array([names, ab, slg]).T
    table2 = pd.DataFrame(table2, columns=['player', 'ab', 'slg'])
    table2 = table2.drop_duplicates('player')

    table2['ab'] = table2['ab'].astype(int)
    table2['slg'] = pd.to_numeric(table2['slg'])

    # arbitrarily filter data so that there must be at least 100 AB
    # for SLG to be reliable
    table2 = table2[table2['ab'] >= 100]
    table2 = table2.set_index('player')
    table2.to_csv('../data/slg_stats_2010.csv')

    return table2


if __name__ == '__main__':
    scrape_positions_2011()
    print 'completed scraping for 2011 position data'

    scrape_slg_2010()
    print 'completed scraping for 2010 slg data'
