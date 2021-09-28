---
layout: post
title: "정규표현식 - 메타문자"
categories: booststudy
tags: plus
comments: true
---
(메타문자) 정규표현식을 사용할 때, 참고하기 위해 작성하였다.

본 내용은 'https://nachwon.github.io/regular-expressions/'을 참고하였으며 보기 편하고 이해하기 쉽게 정리하였다.

## 목차
- [메타 문자](#메타-문자)
- [조건이 있는 표현식](#조건이-있는-표현식)

## 메타 문자
### []\(대괄호) 
- 대괄호 안에 포함된 문자들 중 하나와 매치한다.
- '-' : 대괄호 안에서 '-'는 두 문자 사이의 범위를 나타낸다.
- '^' : 대괄호 안에서 '^'는 반대를 의미한다.
- 자주 사용하는 문자 클래스는 아래의 표와 같다.
<style>
.tablelines table, .tablelines td, .tablelines th {
    border: 1px solid black;
    }
</style>
|문자 클래스|설명|같은 표현|
|\d|숫자|[0-9]|
|\D|숫자가 아닌 것|[^0-9]|
|\w|숫자+문자|[a-zA-Z0-9]|
|\W|숫자+문자가 아닌 것|[^a-zA-Z0-9]|
|\s|공백|[\t\n\r\f\v]|
|\S|공백이 아닌 것|[^\t\n\r\f\v]|
|\b|단어 경계|.|
|\B|단어 경계가 아닌 것|.|
{: .tablelines}

### 기타 메타 문자
<style>
.tablelines table, .tablelines td, .tablelines th {
    border: 1px solid black;
    }
</style>
|문자|의미|예시|
|'.'|모든 문자. '\n'을 제외한 모든 문자와 매칭된다.|a.b -> aab, a0b, abc(x)|
|'*'|앞에 오는 문자가 몇 개이든 모두 매칭('+'와 차이는 앞에 오는 문자가 없는 것도 포함)|ab*c -> ac, abc, abbc, abbbbbc,|
|'+'|앞에 오는 문자가 최소 한 번 이상 반복될 경우 매칭|ab+c -> abc, abbc, abbbc|
|'?'|앞의 문자가 없거나 하나 이상 있을 경우 매칭|ab?c -> ac, abc, abbc(x)|
|{m, n}|앞에 있는 문자가 m번에서 n번까지 반복될 경우 매칭|ab{3,5}c -> abbbc, abbbbc, abbbbbc|
|'\|'|여러 개의 표현 식 중 하나라도 있을 경우 매칭|a\|b\|c\| -> a, b, ac, ab, abc|
|'^'|문자열의 제일 처음과 매칭|^a -> a, aaa, ab, ba(x)|
|'$'|문자열의 제일 마지막과 매칭|a$ -> a, aa, baa, ab(x)|
|'\A'|'^'과 동일하지만 re.MULTILINE 옵션을 무시하고 항상 첫 줄의 시작 문자를 검사|...|
|'\Z'|'$'과 동일하지만 re.MULTILINE 옵션을 무시하고 항상 문자열 마지막 줄의 끝 문자를 검사.|...|
{: .tablelines}

## 조건이 있는 표현식
<style>
.tablelines table, .tablelines td, .tablelines th {
    border: 1px solid black;
    }
</style>
|표현|설명|예시|
|표현식1(?=표현식2)|표현식1 뒤에 문자열이 표현식2와 매칭되면 표현식1 매칭|'Good(?=game)' -> Goodgame(o), Badgame(x), Goodperson(x)|
|표현식1(?!표현식2)|표현식1 뒤의 문자열이 표현식2와 매칭되지 않으면 표현식1 매칭|'Good(?!game)' -> Goodgame(x), Badgame(x), Goodperson(o)|
|(?<=표현식1)표현식2|표현식2 앞의 문자열이 표현식1과 매칭되면 표현식2 매칭|'(?<=Good)game' -> Goodgame(o),  Badgame(x), Goodperson(x)|
|(?<!표현식1)표현식2|표현식2 앞의 문자열이 표현식1과 매칭되지 않으면 표현식2 매칭|'(?<!Good)game' -> Goodgame(x), BadGame(o), Goodperson(x)|
{: .tablelines}

 
