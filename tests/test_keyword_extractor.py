from feature_keyword_extractor.nlp.keyword_extractor import KeywordExtractor


def test_keyword_extractor_uses_only_statistical_text_processing():
    extractor = KeywordExtractor(
        top_k=5,
        stopwords={"的", "了", "和", "后"},
        generic_words={"用户", "系统", "功能", "页面", "按钮"},
        domain_terms={"短信验证码", "登录态", "SIM卡"},
    )

    keywords = extractor.extract(
        [
            "用户输入手机号后，系统发送短信验证码。",
            "短信验证码校验成功后写入登录态，登录态用于后续账号访问。",
            "SIM卡状态异常时需要重新校验手机号。",
        ]
    )

    terms = [item.term for item in keywords]
    assert "短信验证码" in terms
    assert "登录态" in terms
    assert "手机号" in terms
    assert "用户" not in terms
    assert "系统" not in terms
    assert all(item.score > 0 for item in keywords)
