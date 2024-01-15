import pandas as pd
from bs4 import BeautifulSoup

# taken from https://nethackwiki.com/wiki/Monster_difficulty

html_data = """
<thead><tr>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending">Name
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending">Experience
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending">Difficulty
</th></tr></thead><tbody>
<tr>
<td><a href="/wiki/Demogorgon" title="Demogorgon">Demogorgon</a></td>
<td>3297</td>
<td>57
</td></tr>
<tr>
<td><a href="/wiki/Asmodeus" title="Asmodeus">Asmodeus</a></td>
<td>3225</td>
<td>53
</td></tr>
<tr>
<td><a href="/wiki/Baalzebub" title="Baalzebub">Baalzebub</a></td>
<td>2333</td>
<td>45
</td></tr>
<tr>
<td><a href="/wiki/Dispater" title="Dispater">Dispater</a></td>
<td>1923</td>
<td>40
</td></tr>
<tr>
<td><a href="/wiki/Geryon" title="Geryon">Geryon</a></td>
<td>1625</td>
<td>36
</td></tr>
<tr>
<td><a href="/wiki/Orcus" title="Orcus">Orcus</a></td>
<td>1445</td>
<td>36
</td></tr>
<tr>
<td><a href="/wiki/Riders#Pestilence" title="Riders">Pestilence</a></td>
<td>1431</td>
<td>34
</td></tr>
<tr>
<td><a href="/wiki/Riders#Famine" title="Riders">Famine</a></td>
<td>1431</td>
<td>34
</td></tr>
<tr>
<td><a href="/wiki/Riders#Death" title="Riders">Death</a></td>
<td>1431</td>
<td>34
</td></tr>
<tr>
<td><a href="/wiki/Wizard_of_Yendor" title="Wizard of Yendor">Wizard of Yendor</a></td>
<td>1411</td>
<td>34
</td></tr>
<tr>
<td><a href="/wiki/Vlad_the_Impaler" title="Vlad the Impaler">Vlad the Impaler</a></td>
<td>1087</td>
<td>32
</td></tr>
<tr>
<td><a href="/wiki/Master_Kaen" title="Master Kaen">Master Kaen</a></td>
<td>1095</td>
<td>31
</td></tr>
<tr>
<td><a href="/wiki/Yeenoghu" title="Yeenoghu">Yeenoghu</a></td>
<td>1073</td>
<td>31
</td></tr>
<tr>
<td><a href="/wiki/High_priest" title="High priest">high priest</a></td>
<td>1054</td>
<td>30
</td></tr>
<tr>
<td><a href="/wiki/Grand_Master" title="Grand Master">Grand Master</a></td>
<td>1053</td>
<td>30
</td></tr>
<tr>
<td><a href="/wiki/Arch_Priest" title="Arch Priest">Arch Priest</a></td>
<td>876</td>
<td>30
</td></tr>
<tr>
<td><a href="/wiki/Arch-lich" title="Arch-lich">arch-lich</a></td>
<td>915</td>
<td>29
</td></tr>
<tr>
<td><a href="/wiki/Juiblex" title="Juiblex">Juiblex</a></td>
<td>899</td>
<td>26
</td></tr>
<tr>
<td><a href="/wiki/Archon" title="Archon">Archon</a></td>
<td>730</td>
<td>26
</td></tr>
<tr>
<td><a href="/wiki/Mail_daemon" title="Mail daemon">mail daemon</a></td>
<td>1</td>
<td>26
</td></tr>
<tr>
<td><a href="/wiki/Medusa" title="Medusa">Medusa</a></td>
<td>634</td>
<td>25
</td></tr>
<tr>
<td><a href="/wiki/Master_of_Thieves" title="Master of Thieves">Master of Thieves</a></td>
<td>588</td>
<td>24
</td></tr>
<tr>
<td><a href="/wiki/Cyclops" title="Cyclops">Cyclops</a></td>
<td>662</td>
<td>23
</td></tr>
<tr>
<td><a href="/wiki/Neferet_the_Green" title="Neferet the Green">Neferet the Green</a></td>
<td>593</td>
<td>23
</td></tr>
<tr>
<td><a href="/wiki/Norn" title="Norn">Norn</a></td>
<td>588</td>
<td>23
</td></tr>
<tr>
<td><a href="/wiki/Lord_Sato" title="Lord Sato">Lord Sato</a></td>
<td>588</td>
<td>23
</td></tr>
<tr>
<td><a href="/wiki/King_Arthur" title="King Arthur">King Arthur</a></td>
<td>588</td>
<td>23
</td></tr>
<tr>
<td><a href="/wiki/Chromatic_Dragon" title="Chromatic Dragon">Chromatic Dragon</a></td>
<td>586</td>
<td>23
</td></tr>
<tr>
<td><a href="/wiki/Nalzok" title="Nalzok">Nalzok</a></td>
<td>585</td>
<td>23
</td></tr>
<tr>
<td><a href="/wiki/Minion_of_Huhetotl" title="Minion of Huhetotl">Minion of Huhetotl</a></td>
<td>585</td>
<td>23
</td></tr>
<tr>
<td><a href="/wiki/Kraken" title="Kraken">kraken</a></td>
<td>1574</td>
<td>22
</td></tr>
<tr>
<td><a href="/wiki/Croesus" title="Croesus">Croesus</a></td>
<td>746</td>
<td>22
</td></tr>
<tr>
<td><a href="/wiki/Mastodon" title="Mastodon">mastodon</a></td>
<td>611</td>
<td>22
</td></tr>
<tr>
<td><a href="/wiki/Shaman_Karnov" title="Shaman Karnov">Shaman Karnov</a></td>
<td>583</td>
<td>22
</td></tr>
<tr>
<td><a href="/wiki/Pelias" title="Pelias">Pelias</a></td>
<td>583</td>
<td>22
</td></tr>
<tr>
<td><a href="/wiki/Orion" title="Orion">Orion</a></td>
<td>583</td>
<td>22
</td></tr>
<tr>
<td><a href="/wiki/Lord_Carnarvon" title="Lord Carnarvon">Lord Carnarvon</a></td>
<td>583</td>
<td>22
</td></tr>
<tr>
<td><a href="/wiki/Hippocrates" title="Hippocrates">Hippocrates</a></td>
<td>583</td>
<td>22
</td></tr>
<tr>
<td><a href="/wiki/Twoflower" title="Twoflower">Twoflower</a></td>
<td>581</td>
<td>22
</td></tr>
<tr>
<td><a href="/wiki/Thoth_Amon" title="Thoth Amon">Thoth Amon</a></td>
<td>547</td>
<td>22
</td></tr>
<tr>
<td><a href="/wiki/Ixoth" title="Ixoth">Ixoth</a></td>
<td>545</td>
<td>22
</td></tr>
<tr>
<td><a href="/wiki/Iron_golem" title="Iron golem">iron golem</a></td>
<td>545</td>
<td>22
</td></tr>
<tr>
<td><a href="/wiki/Ki-rin" title="Ki-rin">ki-rin</a></td>
<td>552</td>
<td>21
</td></tr>
<tr>
<td><a href="/wiki/Master_lich" title="Master lich">master lich</a></td>
<td>494</td>
<td>21
</td></tr>
<tr>
<td><a href="/wiki/Balrog" title="Balrog">balrog</a></td>
<td>575</td>
<td>20
</td></tr>
<tr>
<td><a href="/wiki/Titan" title="Titan">titan</a></td>
<td>553</td>
<td>20
</td></tr>
<tr>
<td><a href="/wiki/Yellow_dragon" title="Yellow dragon">yellow dragon</a></td>
<td>535</td>
<td>20
</td></tr>
<tr>
<td><a href="/wiki/White_dragon" title="White dragon">white dragon</a></td>
<td>535</td>
<td>20
</td></tr>
<tr>
<td><a href="/wiki/Silver_dragon" title="Silver dragon">silver dragon</a></td>
<td>535</td>
<td>20
</td></tr>
<tr>
<td><a href="/wiki/Red_dragon" title="Red dragon">red dragon</a></td>
<td>535</td>
<td>20
</td></tr>
<tr>
<td><a href="/wiki/Orange_dragon" title="Orange dragon">orange dragon</a></td>
<td>535</td>
<td>20
</td></tr>
<tr>
<td><a href="/wiki/Green_dragon" title="Green dragon">green dragon</a></td>
<td>535</td>
<td>20
</td></tr>
<tr>
<td><a href="/wiki/Gray_dragon" title="Gray dragon">gray dragon</a></td>
<td>535</td>
<td>20
</td></tr>
<tr>
<td><a href="/wiki/Blue_dragon" title="Blue dragon">blue dragon</a></td>
<td>535</td>
<td>20
</td></tr>
<tr>
<td><a href="/wiki/Black_dragon" title="Black dragon">black dragon</a></td>
<td>535</td>
<td>20
</td></tr>
<tr>
<td><a href="/wiki/Master_Assassin" title="Master Assassin">Master Assassin</a></td>
<td>503</td>
<td>20
</td></tr>
<tr>
<td><a href="/wiki/Dark_One" title="Dark One">Dark One</a></td>
<td>498</td>
<td>20
</td></tr>
<tr>
<td><a href="/wiki/Storm_giant" title="Storm giant">storm giant</a></td>
<td>536</td>
<td>19
</td></tr>
<tr>
<td><a href="/wiki/Ashikaga_Takauji" title="Ashikaga Takauji">Ashikaga Takauji</a></td>
<td>488</td>
<td>19
</td></tr>
<tr>
<td><a href="/wiki/Lord_Surtur" title="Lord Surtur">Lord Surtur</a></td>
<td>486</td>
<td>19
</td></tr>
<tr>
<td><a href="/wiki/Angel" title="Angel">Angel</a></td>
<td>485</td>
<td>19
</td></tr>
<tr>
<td><a href="/wiki/Master_mind_flayer" title="Master mind flayer">master mind flayer</a></td>
<td>416</td>
<td>19
</td></tr>
<tr>
<td><a href="/wiki/Jabberwock" title="Jabberwock">jabberwock</a></td>
<td>489</td>
<td>18
</td></tr>
<tr>
<td><a href="/wiki/Glass_golem" title="Glass golem">glass golem</a></td>
<td>409</td>
<td>18
</td></tr>
<tr>
<td><a href="/wiki/Demilich" title="Demilich">demilich</a></td>
<td>376</td>
<td>18
</td></tr>
<tr>
<td><a href="/wiki/Minotaur" title="Minotaur">minotaur</a></td>
<td>504</td>
<td>17
</td></tr>
<tr>
<td><a href="/wiki/Scorpius" title="Scorpius">Scorpius</a></td>
<td>474</td>
<td>17
</td></tr>
<tr>
<td><a href="/wiki/Purple_worm" title="Purple worm">purple worm</a></td>
<td>474</td>
<td>17
</td></tr>
<tr>
<td><a href="/wiki/Nazgul" title="Nazgul">Nazgul</a></td>
<td>376</td>
<td>17
</td></tr>
<tr>
<td><a href="/wiki/Pit_fiend" title="Pit fiend">pit fiend</a></td>
<td>422</td>
<td>16
</td></tr>
<tr>
<td><a href="/wiki/Olog-hai" title="Olog-hai">Olog-hai</a></td>
<td>325</td>
<td>16
</td></tr>
<tr>
<td><a href="/wiki/Guardian_naga" title="Guardian naga">guardian naga</a></td>
<td>295</td>
<td>16
</td></tr>
<tr>
<td><a href="/wiki/Stone_golem" title="Stone golem">stone golem</a></td>
<td>345</td>
<td>15
</td></tr>
<tr>
<td><a href="/wiki/Nalfeshnee" title="Nalfeshnee">nalfeshnee</a></td>
<td>341</td>
<td>15
</td></tr>
<tr>
<td><a href="/wiki/Baluchitherium" title="Baluchitherium">baluchitherium</a></td>
<td>331</td>
<td>15
</td></tr>
<tr>
<td><a href="/wiki/Sandestin" title="Sandestin">sandestin</a></td>
<td>308</td>
<td>15
</td></tr>
<tr>
<td><a href="/wiki/Aligned_priest" title="Aligned priest">aligned priest</a></td>
<td>294</td>
<td>15
</td></tr>
<tr>
<td><a href="/wiki/Shopkeeper" title="Shopkeeper">shopkeeper</a></td>
<td>287</td>
<td>15
</td></tr>
<tr>
<td><a href="/wiki/Vampire_lord" title="Vampire lord">vampire lord</a></td>
<td>399</td>
<td>14
</td></tr>
<tr>
<td><a href="/wiki/Skeleton" title="Skeleton">skeleton</a></td>
<td>359</td>
<td>14
</td></tr>
<tr>
<td><a href="/wiki/Shade" title="Shade">shade</a></td>
<td>357</td>
<td>14
</td></tr>
<tr>
<td><a href="/wiki/Ice_devil" title="Ice devil">ice devil</a></td>
<td>351</td>
<td>14
</td></tr>
<tr>
<td><a href="/wiki/Disenchanter" title="Disenchanter">disenchanter</a></td>
<td>301</td>
<td>14
</td></tr>
<tr>
<td><a href="/wiki/Hell_hound" class="mw-redirect" title="Hell hound">hell hound</a></td>
<td>290</td>
<td>14
</td></tr>
<tr>
<td><a href="/wiki/Guard" title="Guard">guard</a></td>
<td>284</td>
<td>14
</td></tr>
<tr>
<td><a href="/wiki/Captain" class="mw-redirect" title="Captain">captain</a></td>
<td>277</td>
<td>14
</td></tr>
<tr>
<td><a href="/wiki/Prisoner" title="Prisoner">prisoner</a></td>
<td>272</td>
<td>14
</td></tr>
<tr>
<td><a href="/wiki/Trapper" title="Trapper">trapper</a></td>
<td>270</td>
<td>14
</td></tr>
<tr>
<td><a href="/wiki/Lich" title="Lich">lich</a></td>
<td>269</td>
<td>14
</td></tr>
<tr>
<td><a href="/wiki/Frost_giant" title="Frost giant">frost giant</a></td>
<td>296</td>
<td>13
</td></tr>
<tr>
<td><a href="/wiki/Ettin" title="Ettin">ettin</a></td>
<td>291</td>
<td>13
</td></tr>
<tr>
<td><a href="/wiki/Oracle" class="mw-redirect" title="Oracle">Oracle</a></td>
<td>286</td>
<td>13
</td></tr>
<tr>
<td><a href="/wiki/Bone_devil" title="Bone devil">bone devil</a></td>
<td>285</td>
<td>13
</td></tr>
<tr>
<td><a href="/wiki/Baby_yellow_dragon" class="mw-redirect" title="Baby yellow dragon">baby yellow dragon</a></td>
<td>272</td>
<td>13
</td></tr>
<tr>
<td><a href="/wiki/Baby_white_dragon" class="mw-redirect" title="Baby white dragon">baby white dragon</a></td>
<td>272</td>
<td>13
</td></tr>
<tr>
<td><a href="/wiki/Baby_silver_dragon" class="mw-redirect" title="Baby silver dragon">baby silver dragon</a></td>
<td>272</td>
<td>13
</td></tr>
<tr>
<td><a href="/wiki/Baby_red_dragon" class="mw-redirect" title="Baby red dragon">baby red dragon</a></td>
<td>272</td>
<td>13
</td></tr>
<tr>
<td><a href="/wiki/Baby_orange_dragon" class="mw-redirect" title="Baby orange dragon">baby orange dragon</a></td>
<td>272</td>
<td>13
</td></tr>
<tr>
<td><a href="/wiki/Baby_green_dragon" class="mw-redirect" title="Baby green dragon">baby green dragon</a></td>
<td>272</td>
<td>13
</td></tr>
<tr>
<td><a href="/wiki/Baby_gray_dragon" class="mw-redirect" title="Baby gray dragon">baby gray dragon</a></td>
<td>272</td>
<td>13
</td></tr>
<tr>
<td><a href="/wiki/Baby_blue_dragon" class="mw-redirect" title="Baby blue dragon">baby blue dragon</a></td>
<td>272</td>
<td>13
</td></tr>
<tr>
<td><a href="/wiki/Baby_black_dragon" class="mw-redirect" title="Baby black dragon">baby black dragon</a></td>
<td>272</td>
<td>13
</td></tr>
<tr>
<td><a href="/wiki/Titanothere" title="Titanothere">titanothere</a></td>
<td>267</td>
<td>13
</td></tr>
<tr>
<td><a href="/wiki/Mind_flayer" title="Mind flayer">mind flayer</a></td>
<td>263</td>
<td>13
</td></tr>
<tr>
<td><a href="/wiki/Water_troll" title="Water troll">water troll</a></td>
<td>246</td>
<td>13
</td></tr>
<tr>
<td><a href="/wiki/Nurse" title="Nurse">nurse</a></td>
<td>245</td>
<td>13
</td></tr>
<tr>
<td><a href="/wiki/Golden_naga" title="Golden naga">golden naga</a></td>
<td>239</td>
<td>13
</td></tr>
<tr>
<td><a href="/wiki/Vampire" title="Vampire">vampire</a></td>
<td>327</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Aleax" title="Aleax">Aleax</a></td>
<td>298</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Hezrou" title="Hezrou">hezrou</a></td>
<td>267</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Clay_golem" title="Clay golem">clay golem</a></td>
<td>249</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Ghost" title="Ghost">ghost</a></td>
<td>238</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Queen_bee" title="Queen bee">queen bee</a></td>
<td>225</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Watch_captain" class="mw-redirect" title="Watch captain">watch captain</a></td>
<td>221</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Valkyrie_(player_monster)" title="Valkyrie (player monster)">valkyrie</a></td>
<td>221</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Tourist_(player_monster)" title="Tourist (player monster)">tourist</a></td>
<td>221</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Samurai_(player_monster)" title="Samurai (player monster)">samurai</a></td>
<td>221</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Rogue_(player_monster)" title="Rogue (player monster)">rogue</a></td>
<td>221</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Lieutenant" class="mw-redirect" title="Lieutenant">lieutenant</a></td>
<td>221</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Knight_(player_monster)" title="Knight (player monster)">knight</a></td>
<td>221</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Black_pudding" title="Black pudding">black pudding</a></td>
<td>221</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Barbarian_(player_monster)" title="Barbarian (player monster)">barbarian</a></td>
<td>221</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Archeologist_(player_monster)" title="Archeologist (player monster)">archeologist</a></td>
<td>221</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Wizard_(player_monster)" title="Wizard (player monster)">wizard</a></td>
<td>216</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Ranger_(player_monster)" title="Ranger (player monster)">ranger</a></td>
<td>216</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Priest_(player_monster)" title="Priest (player monster)">priestess</a></td>
<td>216</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Priest_(player_monster)" title="Priest (player monster)">priest</a></td>
<td>216</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Healer_(player_monster)" title="Healer (player monster)">healer</a></td>
<td>216</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Elf_(monster)" title="Elf (monster)">elf</a></td>
<td>216</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Cavewoman_(player_monster)" class="mw-redirect" title="Cavewoman (player monster)">cavewoman</a></td>
<td>216</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Caveman_(player_monster)" title="Caveman (player monster)">caveman</a></td>
<td>216</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Lurker_above" title="Lurker above">lurker above</a></td>
<td>214</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Ice_troll" title="Ice troll">ice troll</a></td>
<td>205</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Rock_troll" title="Rock troll">rock troll</a></td>
<td>198</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Umber_hulk" title="Umber hulk">umber hulk</a></td>
<td>194</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Salamander" title="Salamander">salamander</a></td>
<td>159</td>
<td>12
</td></tr>
<tr>
<td><a href="/wiki/Fire_giant" title="Fire giant">fire giant</a></td>
<td>254</td>
<td>11
</td></tr>
<tr>
<td><a href="/wiki/Monk_(player_monster)" title="Monk (player monster)">monk</a></td>
<td>211</td>
<td>11
</td></tr>
<tr>
<td><a href="/wiki/Winged_gargoyle" title="Winged gargoyle">winged gargoyle</a></td>
<td>207</td>
<td>11
</td></tr>
<tr>
<td><a href="/wiki/Water_demon" title="Water demon">water demon</a></td>
<td>196</td>
<td>11
</td></tr>
<tr>
<td><a href="/wiki/Elvenking" title="Elvenking">Elvenking</a></td>
<td>196</td>
<td>11
</td></tr>
<tr>
<td><a href="/wiki/Ogre_king" title="Ogre king">ogre king</a></td>
<td>194</td>
<td>11
</td></tr>
<tr>
<td><a href="/wiki/Doppelganger" title="Doppelganger">doppelganger</a></td>
<td>191</td>
<td>11
</td></tr>
<tr>
<td><a href="/wiki/Zruty" title="Zruty">zruty</a></td>
<td>186</td>
<td>11
</td></tr>
<tr>
<td><a href="/wiki/Giant_mimic" class="mw-redirect" title="Giant mimic">giant mimic</a></td>
<td>186</td>
<td>11
</td></tr>
<tr>
<td><a href="/wiki/Couatl" title="Couatl">couatl</a></td>
<td>180</td>
<td>11
</td></tr>
<tr>
<td><a href="/wiki/Marilith" title="Marilith">marilith</a></td>
<td>177</td>
<td>11
</td></tr>
<tr>
<td><a href="/wiki/Vrock" title="Vrock">vrock</a></td>
<td>176</td>
<td>11
</td></tr>
<tr>
<td><a href="/wiki/Xorn" title="Xorn">xorn</a></td>
<td>139</td>
<td>11
</td></tr>
<tr>
<td><a href="/wiki/Elf-lord" title="Elf-lord">elf-lord</a></td>
<td>123</td>
<td>11
</td></tr>
<tr>
<td><a href="/wiki/Electric_eel" title="Electric eel">electric eel</a></td>
<td>1129</td>
<td>10
</td></tr>
<tr>
<td><a href="/wiki/Flesh_golem" title="Flesh golem">flesh golem</a></td>
<td>186</td>
<td>10
</td></tr>
<tr>
<td><a href="/wiki/Barbed_devil" title="Barbed devil">barbed devil</a></td>
<td>179</td>
<td>10
</td></tr>
<tr>
<td><a href="/wiki/Hill_giant" title="Hill giant">hill giant</a></td>
<td>174</td>
<td>10
</td></tr>
<tr>
<td><a href="/wiki/Erinys" title="Erinys">erinys</a></td>
<td>158</td>
<td>10
</td></tr>
<tr>
<td><a href="/wiki/Fire_vortex" class="mw-redirect" title="Fire vortex">fire vortex</a></td>
<td>142</td>
<td>10
</td></tr>
<tr>
<td><a href="/wiki/Fire_elemental" title="Fire elemental">fire elemental</a></td>
<td>134</td>
<td>10
</td></tr>
<tr>
<td><a href="/wiki/Black_naga" title="Black naga">black naga</a></td>
<td>132</td>
<td>10
</td></tr>
<tr>
<td><a href="/wiki/Water_elemental" title="Water elemental">water elemental</a></td>
<td>126</td>
<td>10
</td></tr>
<tr>
<td><a href="/wiki/Earth_elemental" title="Earth elemental">earth elemental</a></td>
<td>126</td>
<td>10
</td></tr>
<tr>
<td><a href="/wiki/Air_elemental" title="Air elemental">air elemental</a></td>
<td>126</td>
<td>10
</td></tr>
<tr>
<td><a href="/wiki/Sergeant" class="mw-redirect" title="Sergeant">sergeant</a></td>
<td>118</td>
<td>10
</td></tr>
<tr>
<td><a href="/wiki/Giant_mummy" title="Giant mummy">giant mummy</a></td>
<td>116</td>
<td>10
</td></tr>
<tr>
<td><a href="/wiki/Cobra" title="Cobra">cobra</a></td>
<td>90</td>
<td>10
</td></tr>
<tr>
<td><a href="/wiki/Long_worm" title="Long worm">long worm</a></td>
<td>169</td>
<td>10
</td></tr>
<tr>
<td><a href="/wiki/Horned_devil" title="Horned devil">horned devil</a></td>
<td>147</td>
<td>9
</td></tr>
<tr>
<td><a href="/wiki/Xan" title="Xan">xan</a></td>
<td>120</td>
<td>9
</td></tr>
<tr>
<td><a href="/wiki/Wumpus" title="Wumpus">wumpus</a></td>
<td>118</td>
<td>9
</td></tr>
<tr>
<td><a href="/wiki/Stalker" title="Stalker">stalker</a></td>
<td>113</td>
<td>9
</td></tr>
<tr>
<td><a href="/wiki/Large_mimic" class="mw-redirect" title="Large mimic">large mimic</a></td>
<td>113</td>
<td>9
</td></tr>
<tr>
<td><a href="/wiki/Giant_zombie" title="Giant zombie">giant zombie</a></td>
<td>113</td>
<td>9
</td></tr>
<tr>
<td><a href="/wiki/Baby_purple_worm" title="Baby purple worm">baby purple worm</a></td>
<td>113</td>
<td>9
</td></tr>
<tr>
<td><a href="/wiki/Baby_long_worm" class="mw-redirect" title="Baby long worm">baby long worm</a></td>
<td>113</td>
<td>9
</td></tr>
<tr>
<td><a href="/wiki/Steam_vortex" class="mw-redirect" title="Steam vortex">steam vortex</a></td>
<td>112</td>
<td>9
</td></tr>
<tr>
<td><a href="/wiki/Glass_piercer" class="mw-redirect" title="Glass piercer">glass piercer</a></td>
<td>106</td>
<td>9
</td></tr>
<tr>
<td><a href="/wiki/Shark" title="Shark">shark</a></td>
<td>104</td>
<td>9
</td></tr>
<tr>
<td><a href="/wiki/Winter_wolf" class="mw-redirect" title="Winter wolf">winter wolf</a></td>
<td>102</td>
<td>9
</td></tr>
<tr>
<td><a href="/wiki/Hell_hound_pup" class="mw-redirect" title="Hell hound pup">hell hound pup</a></td>
<td>102</td>
<td>9
</td></tr>
<tr>
<td><a href="/wiki/Energy_vortex" class="mw-redirect" title="Energy vortex">energy vortex</a></td>
<td>101</td>
<td>9
</td></tr>
<tr>
<td><a href="/wiki/Warhorse" class="mw-redirect" title="Warhorse">warhorse</a></td>
<td>97</td>
<td>9
</td></tr>
<tr>
<td><a href="/wiki/Troll" title="Troll">troll</a></td>
<td>97</td>
<td>9
</td></tr>
<tr>
<td><a href="/wiki/Ogre_lord" title="Ogre lord">ogre lord</a></td>
<td>97</td>
<td>9
</td></tr>
<tr>
<td><a href="/wiki/Sasquatch" title="Sasquatch">sasquatch</a></td>
<td>95</td>
<td>9
</td></tr>
<tr>
<td><a href="/wiki/Pit_viper" title="Pit viper">pit viper</a></td>
<td>93</td>
<td>9
</td></tr>
<tr>
<td><a href="/wiki/Quantum_mechanic" title="Quantum mechanic">quantum mechanic</a></td>
<td>92</td>
<td>9
</td></tr>
<tr>
<td><a href="/wiki/Mountain_centaur" class="mw-redirect" title="Mountain centaur">mountain centaur</a></td>
<td>88</td>
<td>9
</td></tr>
<tr>
<td><a href="/wiki/Green_slime" title="Green slime">green slime</a></td>
<td>164</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Cockatrice" title="Cockatrice">cockatrice</a></td>
<td>149</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Stone_giant" title="Stone giant">stone giant</a></td>
<td>127</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Giant" class="mw-redirect" title="Giant">giant</a></td>
<td>127</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Succubus" class="mw-redirect" title="Succubus">succubus</a></td>
<td>122</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Incubus" class="mw-redirect" title="Incubus">incubus</a></td>
<td>122</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Wraith" title="Wraith">wraith</a></td>
<td>120</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Djinni" title="Djinni">djinni</a></td>
<td>97</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Gargoyle" title="Gargoyle">gargoyle</a></td>
<td>95</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Wood_golem" title="Wood golem">wood golem</a></td>
<td>92</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Warg" class="mw-redirect" title="Warg">warg</a></td>
<td>92</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Small_mimic" class="mw-redirect" title="Small mimic">small mimic</a></td>
<td>92</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Ettin_mummy" title="Ettin mummy">ettin mummy</a></td>
<td>92</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Shocking_sphere" class="mw-redirect" title="Shocking sphere">shocking sphere</a></td>
<td>91</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Freezing_sphere" class="mw-redirect" title="Freezing sphere">freezing sphere</a></td>
<td>91</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Flaming_sphere" class="mw-redirect" title="Flaming sphere">flaming sphere</a></td>
<td>91</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Ochre_jelly" title="Ochre jelly">ochre jelly</a></td>
<td>88</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Dwarf_king" title="Dwarf king">dwarf king</a></td>
<td>83</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Red_naga" title="Red naga">red naga</a></td>
<td>82</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Python" title="Python">python</a></td>
<td>82</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Pyrolisk" title="Pyrolisk">pyrolisk</a></td>
<td>82</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Watchman" title="Watchman">watchman</a></td>
<td>78</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Soldier" class="mw-redirect" title="Soldier">soldier</a></td>
<td>78</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Grey-elf" title="Grey-elf">Grey-elf</a></td>
<td>78</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Leocrotta" title="Leocrotta">leocrotta</a></td>
<td>76</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Gelatinous_cube" title="Gelatinous cube">gelatinous cube</a></td>
<td>76</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Carnivorous_ape" title="Carnivorous ape">carnivorous ape</a></td>
<td>76</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Tiger" title="Tiger">tiger</a></td>
<td>73</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Guide" title="Guide">guide</a></td>
<td>71</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Apprentice" title="Apprentice">apprentice</a></td>
<td>71</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Acolyte" title="Acolyte">acolyte</a></td>
<td>71</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Rust_monster" title="Rust monster">rust monster</a></td>
<td>70</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Scorpion" title="Scorpion">scorpion</a></td>
<td>67</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Abbot" title="Abbot">abbot</a></td>
<td>66</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Forest_centaur" class="mw-redirect" title="Forest centaur">forest centaur</a></td>
<td>64</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Gremlin" title="Gremlin">gremlin</a></td>
<td>61</td>
<td>8
</td></tr>
<tr>
<td><a href="/wiki/Giant_eel" title="Giant eel">giant eel</a></td>
<td>1075</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Chickatrice" class="mw-redirect" title="Chickatrice">chickatrice</a></td>
<td>136</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Owlbear" title="Owlbear">owlbear</a></td>
<td>94</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Barrow_wight" title="Barrow wight">barrow wight</a></td>
<td>90</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Tengu" title="Tengu">tengu</a></td>
<td>76</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Large_dog" class="mw-redirect" title="Large dog">large dog</a></td>
<td>76</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Large_cat" class="mw-redirect" title="Large cat">large cat</a></td>
<td>76</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Ice_vortex" class="mw-redirect" title="Ice vortex">ice vortex</a></td>
<td>74</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Black_light" title="Black light">black light</a></td>
<td>74</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Leather_golem" title="Leather golem">leather golem</a></td>
<td>73</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Human_mummy" title="Human mummy">human mummy</a></td>
<td>73</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Ettin_zombie" title="Ettin zombie">ettin zombie</a></td>
<td>73</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Elf_mummy" title="Elf mummy">elf mummy</a></td>
<td>73</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Crocodile" title="Crocodile">crocodile</a></td>
<td>73</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Chameleon" title="Chameleon">chameleon</a></td>
<td>73</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Mumak" title="Mumak">mumak</a></td>
<td>68</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Warrior" title="Warrior">warrior</a></td>
<td>66</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Vampire_bat" title="Vampire bat">vampire bat</a></td>
<td>66</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Thug" title="Thug">thug</a></td>
<td>66</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Roshi" title="Roshi">roshi</a></td>
<td>66</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Page" title="Page">page</a></td>
<td>66</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Orc-captain" title="Orc-captain">orc-captain</a></td>
<td>66</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Ninja" title="Ninja">ninja</a></td>
<td>66</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Winter_wolf_cub" class="mw-redirect" title="Winter wolf cub">winter wolf cub</a></td>
<td>64</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Giant_spider" title="Giant spider">giant spider</a></td>
<td>64</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Student" title="Student">student</a></td>
<td>61</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Ogre" title="Ogre">ogre</a></td>
<td>61</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Neanderthal" title="Neanderthal">neanderthal</a></td>
<td>61</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Hunter" title="Hunter">hunter</a></td>
<td>61</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Horse" class="mw-redirect" title="Horse">horse</a></td>
<td>61</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Green-elf" title="Green-elf">Green-elf</a></td>
<td>61</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Chieftain" title="Chieftain">chieftain</a></td>
<td>61</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Attendant" title="Attendant">attendant</a></td>
<td>61</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Yeti" title="Yeti">yeti</a></td>
<td>59</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Panther" title="Panther">panther</a></td>
<td>59</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Lynx" title="Lynx">lynx</a></td>
<td>59</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Werewolf" class="mw-redirect" title="Werewolf">werewolf</a></td>
<td>56</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Water_moccasin" title="Water moccasin">water moccasin</a></td>
<td>48</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Quasit" title="Quasit">quasit</a></td>
<td>36</td>
<td>7
</td></tr>
<tr>
<td><a href="/wiki/Iron_piercer" class="mw-redirect" title="Iron piercer">iron piercer</a></td>
<td>63</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Werewolf" class="mw-redirect" title="Werewolf">werewolf</a></td>
<td>61</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Spotted_jelly" title="Spotted jelly">spotted jelly</a></td>
<td>61</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Gnome_king" title="Gnome king">gnome king</a></td>
<td>61</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Quivering_blob" title="Quivering blob">quivering blob</a></td>
<td>59</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Wolf" class="mw-redirect" title="Wolf">wolf</a></td>
<td>56</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Piranha" title="Piranha">piranha</a></td>
<td>56</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Orc_mummy" title="Orc mummy">orc mummy</a></td>
<td>56</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Lizard" title="Lizard">lizard</a></td>
<td>56</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Gold_golem" title="Gold golem">gold golem</a></td>
<td>56</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Giant_beetle" title="Giant beetle">giant beetle</a></td>
<td>56</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Dwarf_mummy" title="Dwarf mummy">dwarf mummy</a></td>
<td>56</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Brown_pudding" title="Brown pudding">brown pudding</a></td>
<td>56</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Dust_vortex" class="mw-redirect" title="Dust vortex">dust vortex</a></td>
<td>54</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/White_unicorn" class="mw-redirect" title="White unicorn">white unicorn</a></td>
<td>51</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Gray_unicorn" class="mw-redirect" title="Gray unicorn">gray unicorn</a></td>
<td>51</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Dwarf_lord" title="Dwarf lord">dwarf lord</a></td>
<td>51</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Black_unicorn" class="mw-redirect" title="Black unicorn">black unicorn</a></td>
<td>51</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Plains_centaur" class="mw-redirect" title="Plains centaur">plains centaur</a></td>
<td>49</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Snake" title="Snake">snake</a></td>
<td>48</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Woodland-elf" title="Woodland-elf">Woodland-elf</a></td>
<td>46</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Raven" title="Raven">raven</a></td>
<td>46</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Kop_Kaptain" title="Kop Kaptain">Kop Kaptain</a></td>
<td>46</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Rope_golem" title="Rope golem">rope golem</a></td>
<td>44</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Jaguar" title="Jaguar">jaguar</a></td>
<td>44</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Ape" title="Ape">ape</a></td>
<td>41</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Soldier_ant" title="Soldier ant">soldier ant</a></td>
<td>37</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Fire_ant" title="Fire ant">fire ant</a></td>
<td>34</td>
<td>6
</td></tr>
<tr>
<td><a href="/wiki/Blue_jelly" title="Blue jelly">blue jelly</a></td>
<td>45</td>
<td>5
</td></tr>
<tr>
<td><a href="/wiki/Yellow_light" title="Yellow light">yellow light</a></td>
<td>44</td>
<td>5
</td></tr>
<tr>
<td><a href="/wiki/Housecat" class="mw-redirect" title="Housecat">housecat</a></td>
<td>44</td>
<td>5
</td></tr>
<tr>
<td><a href="/wiki/Dog" title="Dog">dog</a></td>
<td>44</td>
<td>5
</td></tr>
<tr>
<td><a href="/wiki/Dingo" class="mw-redirect" title="Dingo">dingo</a></td>
<td>44</td>
<td>5
</td></tr>
<tr>
<td><a href="/wiki/Human_zombie" title="Human zombie">human zombie</a></td>
<td>41</td>
<td>5
</td></tr>
<tr>
<td><a href="/wiki/Gnome_mummy" title="Gnome mummy">gnome mummy</a></td>
<td>41</td>
<td>5
</td></tr>
<tr>
<td><a href="/wiki/Orc_shaman" title="Orc shaman">orc shaman</a></td>
<td>38</td>
<td>5
</td></tr>
<tr>
<td><a href="/wiki/Gnomish_wizard" title="Gnomish wizard">gnomish wizard</a></td>
<td>38</td>
<td>5
</td></tr>
<tr>
<td><a href="/wiki/Violet_fungus" title="Violet fungus">violet fungus</a></td>
<td>34</td>
<td>5
</td></tr>
<tr>
<td><a href="/wiki/Jellyfish" title="Jellyfish">jellyfish</a></td>
<td>34</td>
<td>5
</td></tr>
<tr>
<td><a href="/wiki/Uruk-hai" title="Uruk-hai">Uruk-hai</a></td>
<td>33</td>
<td>5
</td></tr>
<tr>
<td><a href="/wiki/Mordor_orc" title="Mordor orc">Mordor orc</a></td>
<td>33</td>
<td>5
</td></tr>
<tr>
<td><a href="/wiki/Kop_Lieutenant" title="Kop Lieutenant">Kop Lieutenant</a></td>
<td>33</td>
<td>5
</td></tr>
<tr>
<td><a href="/wiki/Bugbear" title="Bugbear">bugbear</a></td>
<td>33</td>
<td>5
</td></tr>
<tr>
<td><a href="/wiki/Killer_bee" title="Killer bee">killer bee</a></td>
<td>31</td>
<td>5
</td></tr>
<tr>
<td><a href="/wiki/Wood_nymph" class="mw-redirect" title="Wood nymph">wood nymph</a></td>
<td>28</td>
<td>5
</td></tr>
<tr>
<td><a href="/wiki/Water_nymph" class="mw-redirect" title="Water nymph">water nymph</a></td>
<td>28</td>
<td>5
</td></tr>
<tr>
<td><a href="/wiki/Mountain_nymph" class="mw-redirect" title="Mountain nymph">mountain nymph</a></td>
<td>28</td>
<td>5
</td></tr>
<tr>
<td><a href="/wiki/Lemure" title="Lemure">lemure</a></td>
<td>28</td>
<td>5
</td></tr>
<tr>
<td><a href="/wiki/Ghoul" title="Ghoul">ghoul</a></td>
<td>28</td>
<td>5
</td></tr>
<tr>
<td><a href="/wiki/Leprechaun" title="Leprechaun">leprechaun</a></td>
<td>59</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Fog_cloud" class="mw-redirect" title="Fog cloud">fog cloud</a></td>
<td>38</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Woodchuck" title="Woodchuck">woodchuck</a></td>
<td>35</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Rock_mole" title="Rock mole">rock mole</a></td>
<td>35</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Imp" title="Imp">imp</a></td>
<td>33</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Gnome_lord" title="Gnome lord">gnome lord</a></td>
<td>33</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Pony" title="Pony">pony</a></td>
<td>31</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Straw_golem" title="Straw golem">straw golem</a></td>
<td>28</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Rock_piercer" class="mw-redirect" title="Rock piercer">rock piercer</a></td>
<td>28</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Red_naga_hatchling" title="Red naga hatchling">red naga hatchling</a></td>
<td>28</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Paper_golem" title="Paper golem">paper golem</a></td>
<td>28</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Kobold_mummy" title="Kobold mummy">kobold mummy</a></td>
<td>28</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Guardian_naga_hatchling" title="Guardian naga hatchling">guardian naga hatchling</a></td>
<td>28</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Gray_ooze" title="Gray ooze">gray ooze</a></td>
<td>28</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Golden_naga_hatchling" title="Golden naga hatchling">golden naga hatchling</a></td>
<td>28</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Elf_zombie" title="Elf zombie">elf zombie</a></td>
<td>28</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Black_naga_hatchling" title="Black naga hatchling">black naga hatchling</a></td>
<td>28</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Baby_crocodile" class="mw-redirect" title="Baby crocodile">baby crocodile</a></td>
<td>28</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Kobold_shaman" title="Kobold shaman">kobold shaman</a></td>
<td>27</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Kop_Sergeant" title="Kop Sergeant">Kop Sergeant</a></td>
<td>22</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Hill_orc" title="Hill orc">hill orc</a></td>
<td>22</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Dwarf_(monster)" title="Dwarf (monster)">dwarf</a></td>
<td>22</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Giant_ant" title="Giant ant">giant ant</a></td>
<td>20</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Centipede" title="Centipede">centipede</a></td>
<td>19</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Wererat" class="mw-redirect" title="Wererat">wererat</a></td>
<td>17</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Werejackal" class="mw-redirect" title="Werejackal">werejackal</a></td>
<td>17</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Rothe" title="Rothe">rothe</a></td>
<td>17</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Rabid_rat" class="mw-redirect" title="Rabid rat">rabid rat</a></td>
<td>17</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Monkey" title="Monkey">monkey</a></td>
<td>17</td>
<td>4
</td></tr>
<tr>
<td><a href="/wiki/Wererat" class="mw-redirect" title="Wererat">wererat</a></td>
<td>22</td>
<td>3
</td></tr>
<tr>
<td><a href="/wiki/Werejackal" class="mw-redirect" title="Werejackal">werejackal</a></td>
<td>22</td>
<td>3
</td></tr>
<tr>
<td><a href="/wiki/Kobold_lord" title="Kobold lord">kobold lord</a></td>
<td>22</td>
<td>3
</td></tr>
<tr>
<td><a href="/wiki/Giant_bat" title="Giant bat">giant bat</a></td>
<td>22</td>
<td>3
</td></tr>
<tr>
<td><a href="/wiki/Little_dog" class="mw-redirect" title="Little dog">little dog</a></td>
<td>20</td>
<td>3
</td></tr>
<tr>
<td><a href="/wiki/Kitten" class="mw-redirect" title="Kitten">kitten</a></td>
<td>20</td>
<td>3
</td></tr>
<tr>
<td><a href="/wiki/Homunculus" title="Homunculus">homunculus</a></td>
<td>19</td>
<td>3
</td></tr>
<tr>
<td><a href="/wiki/Orc_zombie" title="Orc zombie">orc zombie</a></td>
<td>17</td>
<td>3
</td></tr>
<tr>
<td><a href="/wiki/Iguana" title="Iguana">iguana</a></td>
<td>17</td>
<td>3
</td></tr>
<tr>
<td><a href="/wiki/Floating_eye" title="Floating eye">floating eye</a></td>
<td>17</td>
<td>3
</td></tr>
<tr>
<td><a href="/wiki/Dwarf_zombie" title="Dwarf zombie">dwarf zombie</a></td>
<td>17</td>
<td>3
</td></tr>
<tr>
<td><a href="/wiki/Orc" title="Orc">orc</a></td>
<td>13</td>
<td>3
</td></tr>
<tr>
<td><a href="/wiki/Keystone_Kop" title="Keystone Kop">Keystone Kop</a></td>
<td>13</td>
<td>3
</td></tr>
<tr>
<td><a href="/wiki/Hobgoblin" title="Hobgoblin">hobgoblin</a></td>
<td>13</td>
<td>3
</td></tr>
<tr>
<td><a href="/wiki/Gnome_(monster)" title="Gnome (monster)">gnome</a></td>
<td>13</td>
<td>3
</td></tr>
<tr>
<td><a href="/wiki/Manes" title="Manes">manes</a></td>
<td>8</td>
<td>3
</td></tr>
<tr>
<td><a href="/wiki/Garter_snake" title="Garter snake">garter snake</a></td>
<td>8</td>
<td>3
</td></tr>
<tr>
<td><a href="/wiki/Cave_spider" title="Cave spider">cave spider</a></td>
<td>8</td>
<td>3
</td></tr>
<tr>
<td><a href="/wiki/Shrieker" title="Shrieker">shrieker</a></td>
<td>28</td>
<td>2
</td></tr>
<tr>
<td><a href="/wiki/Large_kobold" title="Large kobold">large kobold</a></td>
<td>13</td>
<td>2
</td></tr>
<tr>
<td><a href="/wiki/Hobbit" title="Hobbit">hobbit</a></td>
<td>13</td>
<td>2
</td></tr>
<tr>
<td><a href="/wiki/Gas_spore" title="Gas spore">gas spore</a></td>
<td>12</td>
<td>2
</td></tr>
<tr>
<td><a href="/wiki/Red_mold" class="mw-redirect" title="Red mold">red mold</a></td>
<td>9</td>
<td>2
</td></tr>
<tr>
<td><a href="/wiki/Green_mold" class="mw-redirect" title="Green mold">green mold</a></td>
<td>9</td>
<td>2
</td></tr>
<tr>
<td><a href="/wiki/Brown_mold" class="mw-redirect" title="Brown mold">brown mold</a></td>
<td>9</td>
<td>2
</td></tr>
<tr>
<td><a href="/wiki/Acid_blob" title="Acid blob">acid blob</a></td>
<td>9</td>
<td>2
</td></tr>
<tr>
<td><a href="/wiki/Yellow_mold" class="mw-redirect" title="Yellow mold">yellow mold</a></td>
<td>8</td>
<td>2
</td></tr>
<tr>
<td><a href="/wiki/Gnome_zombie" title="Gnome zombie">gnome zombie</a></td>
<td>8</td>
<td>2
</td></tr>
<tr>
<td><a href="/wiki/Giant_rat" class="mw-redirect" title="Giant rat">giant rat</a></td>
<td>8</td>
<td>2
</td></tr>
<tr>
<td><a href="/wiki/Gecko" title="Gecko">gecko</a></td>
<td>8</td>
<td>2
</td></tr>
<tr>
<td><a href="/wiki/Coyote" class="mw-redirect" title="Coyote">coyote</a></td>
<td>8</td>
<td>2
</td></tr>
<tr>
<td><a href="/wiki/Human_(monster)" title="Human (monster)">human</a></td>
<td>6</td>
<td>2
</td></tr>
<tr>
<td><a href="/wiki/Bat" title="Bat">bat</a></td>
<td>6</td>
<td>2
</td></tr>
<tr>
<td><a href="/wiki/Kobold" title="Kobold">kobold</a></td>
<td>6</td>
<td>1
</td></tr>
<tr>
<td><a href="/wiki/Goblin" title="Goblin">goblin</a></td>
<td>6</td>
<td>1
</td></tr>
<tr>
<td><a href="/wiki/Lichen" title="Lichen">lichen</a></td>
<td>4</td>
<td>1
</td></tr>
<tr>
<td><a href="/wiki/Fox" class="mw-redirect" title="Fox">fox</a></td>
<td>4</td>
<td>1
</td></tr>
<tr>
<td><a href="/wiki/Sewer_rat" class="mw-redirect" title="Sewer rat">sewer rat</a></td>
<td>1</td>
<td>1
</td></tr>
<tr>
<td><a href="/wiki/Newt" title="Newt">newt</a></td>
<td>1</td>
<td>1
</td></tr>
<tr>
<td><a href="/wiki/Kobold_zombie" title="Kobold zombie">kobold zombie</a></td>
<td>1</td>
<td>1
</td></tr>
<tr>
<td><a href="/wiki/Jackal" class="mw-redirect" title="Jackal">jackal</a></td>
<td>1</td>
<td>1
</td></tr>
<tr>
<td><a href="/wiki/Grid_bug" title="Grid bug">grid bug</a></td>
<td>1</td>
<td>1
</td></tr></tbody><tfoot></tfoot>

"""

soup = BeautifulSoup(html_data, "html.parser")

table = soup.find("tbody")

monster_data = []

for row in table.find_all("tr"):
    columns = row.find_all("td")

    if columns:  # Exclude header row
        name = columns[0].find("a").text.strip()
        experience = int(columns[1].text.strip())
        difficulty = int(columns[2].text.strip())

        monster_data.append(
            {
                "Name": name,
                "Experience": experience,
                "Difficulty": difficulty,
            }
        )

monster_data = pd.DataFrame(monster_data)

# taken from https://www.steelypips.org/nethack/experience-spoiler.html

monster_level = {
    "kobold zombie": 0,
    "mail daemon": 25,
    "newt": 0,
    "long worm tail": 0,
    "jackal": 0,
    "fox": 0,
    "sewer rat": 0,
    "grid bug": 0,
    "lichen": 0,
    "kobold": 0,
    "goblin": 0,
    "bat": 0,
    "manes": 1,
    "giant rat": 1,
    "yellow mold": 1,
    "garter snake": 1,
    "gnome zombie": 1,
    "gecko": 1,
    "acid blob": 1,
    "brown mold": 1,
    "green mold": 1,
    "red mold": 1,
    "human": 0,
    "coyote": 1,
    "cave spider": 1,
    "gas spore": 1,
    "hobbit": 1,
    "large kobold": 1,
    "hobgoblin": 1,
    "orc": 1,
    "gnome": 1,
    "Keystone Kop": 1,
    "floating eye": 2,
    "rothe": 2,
    "orc zombie": 2,
    "dwarf zombie": 2,
    "iguana": 2,
    "killer bee": 1,
    "centipede": 2,
    "werejackal": 2,
    "rabid rat": 2,
    "wererat": 2,
    "monkey": 2,
    "giant ant": 2,
    "little dog": 2,
    "kitten": 2,
    "dwarf": 2,
    "homunculus": 2,
    "kobold lord": 2,
    "hill orc": 2,
    "giant bat": 2,
    "Kop Sergeant": 2,
    "wererat": 2,
    "werejackal": 2,
    "kobold shaman": 2,
    "lemure": 3,
    "rock piercer": 3,
    "shrieker": 3,
    "kobold mummy": 3,
    "red naga hatchling": 3,
    "black naga hatchling": 3,
    "golden naga hatchling": 3,
    "guardian naga hatchling": 3,
    "gray ooze": 3,
    "elf zombie": 3,
    "ghoul": 3,
    "baby crocodile": 3,
    "rock mole": 3,
    "woodchuck": 3,
    "wood nymph": 3,
    "water nymph": 3,
    "mountain nymph": 3,
    "pony": 3,
    "straw golem": 3,
    "paper golem": 3,
    "imp": 3,
    "quasit": 3,
    "fog cloud": 3,
    "bugbear": 3,
    "Mordor orc": 3,
    "Uruk-hai": 3,
    "gnome lord": 3,
    "Kop Lieutenant": 3,
    "violet fungus": 3,
    "jellyfish": 3,
    "fire ant": 3,
    "orc shaman": 3,
    "yellow light": 3,
    "gnomish wizard": 3,
    "soldier ant": 3,
    "gnome mummy": 4,
    "human zombie": 4,
    "dog": 4,
    "dingo": 4,
    "housecat": 4,
    "jaguar": 4,
    "ape": 4,
    "rope golem": 4,
    "blue jelly": 4,
    "raven": 4,
    "white unicorn": 4,
    "gray unicorn": 4,
    "black unicorn": 4,
    "snake": 4,
    "water moccasin": 4,
    "Kop Kaptain": 4,
    "Woodland-elf": 4,
    "dust vortex": 4,
    "dwarf lord": 4,
    "plains centaur": 4,
    "giant beetle": 5,
    "orc mummy": 5,
    "dwarf mummy": 5,
    "brown pudding": 5,
    "gold golem": 5,
    "lizard": 5,
    "iron piercer": 5,
    "quivering blob": 5,
    "wolf": 5,
    "werewolf": 5,
    "lynx": 5,
    "panther": 5,
    "leprechaun": 5,
    "yeti": 5,
    "piranha": 5,
    "gremlin": 5,
    "spotted jelly": 5,
    "horse": 5,
    "gnome king": 5,
    "ogre": 5,
    "mumak": 5,
    "giant spider": 5,
    "werewolf": 5,
    "Green-elf": 5,
    "student": 5,
    "chieftain": 5,
    "neanderthal": 5,
    "attendant": 5,
    "hunter": 5,
    "orc-captain": 5,
    "vampire bat": 5,
    "forest centaur": 5,
    "winter wolf cub": 5,
    "scorpion": 5,
    "black light": 5,
    "rust monster": 5,
    "page": 5,
    "abbot": 5,
    "thug": 5,
    "ninja": 5,
    "roshi": 5,
    "warrior": 5,
    "ice vortex": 5,
    "ettin zombie": 6,
    "leather golem": 6,
    "chameleon": 6,
    "crocodile": 6,
    "acolyte": 5,
    "guide": 5,
    "apprentice": 5,
    "gargoyle": 6,
    "gelatinous cube": 6,
    "large dog": 6,
    "large cat": 6,
    "tiger": 6,
    "tengu": 6,
    "elf mummy": 6,
    "human mummy": 6,
    "leocrotta": 6,
    "soldier": 6,
    "watchman": 6,
    "carnivorous ape": 6,
    "Grey-elf": 6,
    "pyrolisk": 6,
    "python": 6,
    "dwarf king": 6,
    "mountain centaur": 6,
    "red naga": 6,
    "ochre jelly": 6,
    "cobra": 6,
    "pit viper": 6,
    "freezing sphere": 6,
    "flaming sphere": 6,
    "shocking sphere": 6,
    "small mimic": 7,
    "wood golem": 7,
    "barrow wight": 3,
    "warg": 7,
    "ettin mummy": 7,
    "quantum mechanic": 7,
    "sasquatch": 7,
    "warhorse": 7,
    "energy vortex": 6,
    "owlbear": 5,
    "glass piercer": 7,
    "ogre lord": 7,
    "troll": 7,
    "djinni": 7,
    "xan": 7,
    "shark": 7,
    "winter wolf": 7,
    "hell hound pup": 7,
    "steam vortex": 7,
    "large mimic": 8,
    "baby long worm": 8,
    "baby purple worm": 8,
    "giant zombie": 8,
    "wumpus": 8,
    "stalker": 8,
    "giant mummy": 8,
    "sergeant": 8,
    "succubus": 6,
    "incubus": 6,
    "giant": 6,
    "stone giant": 6,
    "air elemental": 8,
    "earth elemental": 8,
    "water elemental": 8,
    "wraith": 6,
    "xorn": 8,
    "horned devil": 6,
    "elf-lord": 8,
    "black naga": 8,
    "fire elemental": 8,
    "chickatrice": 4,
    "fire vortex": 8,
    "salamander": 8,
    "cockatrice": 5,
    "marilith": 7,
    "erinys": 7,
    "green slime": 6,
    "long worm": 8,
    "vrock": 8,
    "hill giant": 8,
    "barbed devil": 8,
    "water demon": 8,
    "couatl": 8,
    "giant mimic": 9,
    "zruty": 9,
    "flesh golem": 9,
    "umber hulk": 9,
    "winged gargoyle": 9,
    "ogre king": 9,
    "doppelganger": 9,
    "rock troll": 9,
    "Elvenking": 9,
    "ice troll": 9,
    "queen bee": 9,
    "lurker above": 10,
    "monk": 10,
    "ghost": 10,
    "elf": 10,
    "caveman": 10,
    "cavewoman": 10,
    "healer": 10,
    "priest": 10,
    "priestess": 10,
    "ranger": 10,
    "wizard": 10,
    "black pudding": 10,
    "lieutenant": 10,
    "watch captain": 10,
    "archeologist": 10,
    "barbarian": 10,
    "knight": 10,
    "rogue": 10,
    "samurai": 10,
    "tourist": 10,
    "valkyrie": 10,
    "golden naga": 10,
    "nurse": 11,
    "water troll": 11,
    "clay golem": 11,
    "hezrou": 9,
    "fire giant": 9,
    "lich": 11,
    "mind flayer": 9,
    "baby gray dragon": 12,
    "baby silver dragon": 12,
    "baby red dragon": 12,
    "baby white dragon": 12,
    "baby orange dragon": 12,
    "baby black dragon": 12,
    "baby blue dragon": 12,
    "baby green dragon": 12,
    "baby yellow dragon": 12,
    "titanothere": 12,
    "trapper": 12,
    "bone devil": 9,
    "disenchanter": 12,
    "prisoner": 12,
    "captain": 12,
    "Oracle": 12,
    "shopkeeper": 12,
    "hell hound": 12,
    "guard": 12,
    "guardian naga": 12,
    "Aleax": 10,
    "ettin": 10,
    "aligned priest": 12,
    "frost giant": 10,
    "Olog-hai": 13,
    "sandestin": 13,
    "vampire": 10,
    "nalfeshnee": 11,
    "ice devil": 11,
    "baluchitherium": 14,
    "stone golem": 14,
    "shade": 12,
    "skeleton": 12,
    "demilich": 14,
    "Nazgul": 13,
    "vampire lord": 12,
    "glass golem": 16,
    "pit fiend": 13,
    "master mind flayer": 13,
    "Angel": 14,
    "purple worm": 15,
    "master lich": 17,
    "jabberwock": 15,
    "Vlad the Impaler": 14,
    "Scorpius": 15,
    "Ashikaga Takauji": 15,
    "Lord Surtur": 15,
    "Dark One": 15,
    "Master Assassin": 15,
    "minotaur": 15,
    "gray dragon": 15,
    "silver dragon": 15,
    "red dragon": 15,
    "white dragon": 15,
    "orange dragon": 15,
    "black dragon": 15,
    "blue dragon": 15,
    "green dragon": 15,
    "yellow dragon": 15,
    "ki-rin": 16,
    "Ixoth": 15,
    "titan": 16,
    "storm giant": 16,
    "Thoth Amon": 16,
    "iron golem": 18,
    "balrog": 16,
    "Minion of Huhetotl": 16,
    "Nalzok": 16,
    "Lord Carnarvon": 20,
    "Pelias": 20,
    "Shaman Karnov": 20,
    "Hippocrates": 20,
    "Orion": 20,
    "Chromatic Dragon": 16,
    "Twoflower": 20,
    "King Arthur": 20,
    "Master of Thieves": 20,
    "Lord Sato": 20,
    "Norn": 20,
    "Neferet the Green": 20,
    "mastodon": 20,
    "Medusa": 20,
    "Cyclops": 18,
    "Archon": 19,
    "Croesus": 20,
    "Juiblex": 22,
    "Arch Priest": 25,
    "arch-lich": 25,
    "Grand Master": 25,
    "Yeenoghu": 25,
    "high priest": 25,
    "giant eel": 5,
    "Master Kaen": 25,
    "electric eel": 7,
    "Wizard of Yendor": 30,
    "Death": 30,
    "Pestilence": 30,
    "Famine": 30,
    "Orcus": 30,
    "kraken": 20,
    "Geryon": 33,
    "Dispater": 36,
    "Baalzebub": 41,
    "Asmodeus": 49,
    "Demogorgon": 50,
}

monster_data["Level"] = monster_data["Name"].map(monster_level)

monster_data.to_csv("monster_data.csv")
