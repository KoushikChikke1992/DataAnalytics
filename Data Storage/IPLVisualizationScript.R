####nessesscary libraries#########
library(tidyr)
library("stringr")
library(RODBC)
library(car)
library(dplyr)
library(ggplot2)
####connection string#############
conn <- odbcDriverConnect('driver={SQL Server};server=SANTHOSH\\SQLEXPRESS;database=IPL_DW;trusted_connection=true')
###############################batting average versus player age###################################
####sql query#######
agePlayerBattingAverage = sqlQuery(conn, "SELECT pd.playerAge,pf.playerKey,pf.tournamentKey,pf.battingAverage
                    FROM player_Dim pd
                    LEFT JOIN playerTournamentPerformance_Fact pf
                    ON pf.playerKey=pd.playerKey
                    ")
agePlayerBattingAverage<-agePlayerBattingAverage %>% replace_na(list(playerAge=0,battingAverage=0))
scatterplot(agePlayerBattingAverage$battingAverage ~ agePlayerBattingAverage$playerAge, data = agePlayerBattingAverage,xlab = "age",ylab = "batting average",
            smoother = FALSE, grid = FALSE, frame = FALSE)

##############################batting and bowling performance every ipl tournament#####################
tournamentPerformance = sqlQuery(conn, "SELECT td.tournamentName,pf.runsScored,pf.wicketsTaken,pf.playerKey,pf.tournamentKey
                    FROM tournament_Dim td
                    LEFT JOIN playerTournamentPerformance_Fact pf
                    ON pf.tournamentKey=td.tournamentKey
                    ")
tournamentrPerformance<-tournamentPerformance %>% replace_na(list(runsScored=0,wicketsTaken=0))
only2016<-subset(tournamentrPerformance,tournamentrPerformance$tournamentName=="IPL 2016")
only2017<-subset(tournamentrPerformance,tournamentrPerformance$tournamentName=="IPL 2017")
only2018<-subset(tournamentrPerformance,tournamentrPerformance$tournamentName=="IPL 2018")
only2019<-subset(tournamentrPerformance,tournamentrPerformance$tournamentName=="IPL 2019")
runScoredValues<-c(sum(only2016$runsScored),sum(only2017$runsScored),sum(only2018$runsScored),sum(only2019$runsScored))
wicketsTaken<-c(sum(only2016$wicketsTaken),sum(only2017$wicketsTaken),sum(only2018$wicketsTaken),sum(only2019$wicketsTaken))
par(mfrow=c(2,1))
plot(runScoredValues,type="o",col="red",xlab="2016 to 2019 ",ylab="overall runs Scored by players",main = "runscored vs ipl seasons")
plot(wicketsTaken,type="o",col="blue",xlab="2016 to 2019 ",ylab="overall wickets taken",main = "wickets taken vs ipl seasons")
############################wickets taken by all teams###################################
#bar graph
iplTeam = sqlQuery(conn, "SELECT td.teamName,pf.wicketsTaken,pf.playerKey,pf.tournamentKey
                    FROM team_Dim td
                    LEFT JOIN playerTournamentPerformance_Fact pf
                    ON pf.teamKey=td.teamKey
                    ")
#replace all the NA with 0
newiplTeam<-iplTeam %>% replace_na(list(wicketsTaken=0))
#use aggregate function to calculate the total wickets based on teams
aggResult<-aggregate(newiplTeam$wicketsTaken,by=list(Category=newiplTeam$teamName),FUN=sum)
#use spread to make the data horizontal
updatedIPLwickets<-spread(aggResult,"Category","x",1:2,fill = NA) 
#plot bar graph
values<-c(updatedIPLwickets$`Chennai Super Kings`,updatedIPLwickets$`Delhi Daredevils`,
          updatedIPLwickets$`Gujarat Lions`,updatedIPLwickets$`Kings XI Punjab`,
          updatedIPLwickets$`Kolkata Knight Riders`,updatedIPLwickets$`Mumbai Indians`,
          updatedIPLwickets$`Rajasthan Royals`,updatedIPLwickets$`Rising Pune Supergiant`,
          updatedIPLwickets$`Royal Challengers Bangalore`,updatedIPLwickets$`Sunrisers Hyderabad`)
names<-c("CSK","DD","GL","KIP","KKR","MI","RR","RPS","RCB","SH")
barplot(values,names.arg=names,xlab="Teams",ylab="wickets taken",col="blue",main="overall wickets taken",border="red")
#######################number of Half centuries pie chart######################
playerTeamHalfCenturies = sqlQuery(conn, "SELECT td.teamName,pf.numberOfHalfCenturies,pf.playerKey,pf.tournamentKey
                    FROM team_Dim td
                    LEFT JOIN playerTournamentPerformance_Fact pf
                    ON pf.teamKey=td.teamKey
                    ")
newplayerTeamHalfCenturies<-playerTeamHalfCenturies %>% replace_na(list(numberOfHalfCenturies=0))
teamNumberOfCenturies<-aggregate(newplayerTeamHalfCenturies$numberOfHalfCenturies,by=list(Category=newplayerTeamHalfCenturies$teamName),FUN=sum)
teamNumberOfHalfCenturiesHoriz<-spread(teamNumberOfCenturies,"Category","x",1:2,fill = NA) 
slices<-c(teamNumberOfHalfCenturiesHoriz$`Chennai Super Kings`,teamNumberOfHalfCenturiesHoriz$`Delhi Daredevils`,
          teamNumberOfHalfCenturiesHoriz$`Gujarat Lions`,teamNumberOfHalfCenturiesHoriz$`Kings XI Punjab`,
          teamNumberOfHalfCenturiesHoriz$`Kolkata Knight Riders`,teamNumberOfHalfCenturiesHoriz$`Mumbai Indians`,
          teamNumberOfHalfCenturiesHoriz$`Rajasthan Royals`,teamNumberOfHalfCenturiesHoriz$`Rising Pune Supergiant`,
          teamNumberOfHalfCenturiesHoriz$`Royal Challengers Bangalore`,teamNumberOfHalfCenturiesHoriz$`Sunrisers Hyderabad`)
lbls<-c("CSK","DD","GL","KIP","KKR","MI","RR","RPS","RCB","SH")
pct<-round(slices/sum(slices)*100)
lbls<-paste(lbls,pct)
lbls<-paste(lbls,"%",sep="")
par(mfrow=c(1,1))
pie(slices,labels=lbls,col=rainbow(length(lbls)),main="pie chart to show number of Half centuries by each team from 2016 to 2019" )







