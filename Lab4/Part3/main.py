from summa.summarizer import summarize
from summa import keywords

text = "Neo-Nazism consists of post-World War II militant social or political movements seeking to revive and implement the ideology of Nazism. Neo-Nazis seek to employ their ideology to promote hatred and attack minorities, or in some cases to create a fascist political state. It is a global phenomenon, with organized representation in many countries and international networks. It borrows elements from Nazi doctrine, including ultranationalism, racism, xenophobia, ableism, homophobia, anti-Romanyism, antisemitism, anti-communism and initiating the Fourth Reich. Holocaust denial is a common feature, as is the incorporation of Nazi symbols and admiration of Adolf Hitler. In some European and Latin American countries, laws prohibit the expression of pro-Nazi, racist, anti-Semitic, or homophobic views. Many Nazi-related symbols are banned in European countries (especially Germany) in an effort to curtail neo-Nazism. The term neo-Nazism describes any post-World War II militant, social or political movements seeking to revive the ideology of Nazism in whole or in part. The term neo-Nazism can also refer to the ideology of these movements, which may borrow elements from Nazi doctrine, including ultranationalism, anti-communism, racism, ableism, xenophobia, homophobia, anti-Romanyism, antisemitism, up to initiating the Fourth Reich. Holocaust denial is a common feature, as is the incorporation of Nazi symbols and admiration of Adolf Hitler. Neo-Nazism is considered a particular form of far-right politics and right-wing extremism."
news_article = 'Tidigt på söndagsmorgonen skakades Ljura av en explosion som orsakade stora skador på en fastighet och skadade en man i 70-årsåldern. Senare på kvällen inträffade ytterligare en sprängning i Sandbyhov. Under måndagen meddelade polisen att tre personer gripits. Två män misstänks för inblandning i sprängningen i Sandbyhov. Den tredje är en pojke under 15 år som greps i Lindö, där han enligt polisen togs på bar gärning med ett föremål som misstänks vara farligt. På tisdagen är polisen fortsatt förtegen om utredningen, som leds av åklagare Anna Hjorth. – Vi jobbar på för att få klarhet i alla de här sprängningarna och vi jobbar för fullt för att nya sprängningar inte ska äga rum. Det är fasansfulla brott som drabbar många, säger kriminalkommissarie Jan Staaf vid Norrköpingspolisen. Han vill inte gå in på vilka utredningsåtgärder som genomförs. – Men oavsett om det handlar om ett mord, ett mordförsök eller en sprängning med allmänfarlig ödeläggelse handlar det om att bringa klarhet i vad som hänt och fånga upp de spår och vittnen som finns på platsen. Nu har jag sagt vad jag kan, tror jag. Vi har sökt Anna Hjorth för en kommentar kring utredningen och om misstankarna mot de gripna kvarstår. Hon återkopplar via mejl på tisdagen: "Det stämmer att jag är förundersökningsledare i ett par ärenden från söndagen. Jag kan dock inte lämna några uppgifter eller kommentarer i nuläget, mer än att sedvanliga utredningsåtgärder med bland annat förhör, informationsinhämtning och olika tekniska utredningar pågår”, skriver hon.'
chat = 'Core programming practices form the foundation of reliable, maintainable, and scalable software development. They guide programmers in writing code that is clear, predictable, and adaptable as systems grow. One of the most important practices is writing clean and readable code, which means choosing meaningful variable names, using consistent formatting, and breaking complex logic into smaller, understandable parts. Readability reduces cognitive load and helps new developers quickly understand the purpose and flow of the code. Another essential practice is modular programming, which divides a program into separate functions or modules, each responsible for a single task. This separation makes it easier to test, debug, and reuse code. Within modular programming, baseline functions serve as the program’s fundamental building blocks. These are simple, well-defined functions that perform essential operations such as data validation, input handling, error checking, or formatting output. Baseline functions act as reliable anchors that more complex behaviors can build upon. Good programming also emphasizes error handling and defensive coding. Anticipating potential failures—such as invalid input, missing files, or network issues—ensures that programs behave gracefully even under unexpected conditions. Equally important is version control usage, which provides a history of changes, supports collaboration, and allows developers to revert to earlier states when needed. Testing is another core practice. Unit tests verify that individual functions work correctly, while integration tests ensure that modules interact as expected. Baseline functions especially benefit from thorough testing because they are reused widely and form the program’s internal backbone. Finally, programmers must adopt continuous improvement habits: refactoring code to improve efficiency, documenting important functionality, and staying updated with evolving tools and standards. Together, these practices help developers create software that is robust, understandable, and ready for long-term success.'

print("------------ Test Text -----------")
print(summarize(text, ratio=0.2), "\n\n")

print(summarize(text, words=50), "\n")

print("keywords: \n", keywords.keywords(text), "\n")

print("Top 3 Keywords:\n",keywords.keywords(text,words=3), "\n")

print("------------ News Article -----------")
print(summarize(news_article, ratio=0.2), "\n\n")

print(summarize(news_article, words=50), "\n")

print("keywords: \n", keywords.keywords(news_article), "\n")

print("Top 3 Keywords:\n",keywords.keywords(news_article,words=3), "\n")

print("------------ Chat -----------")
print(summarize(chat, ratio=0.2), "\n\n")

print(summarize(chat, words=50), "\n")

print("keywords: \n", keywords.keywords(chat), "\n")

print("Top 3 Keywords:\n",keywords.keywords(chat,words=3), "\n")