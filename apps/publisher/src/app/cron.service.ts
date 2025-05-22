import { HttpException,Injectable} from '@nestjs/common';
import { Cron} from '@nestjs/schedule';
import { AppService } from './app.service';
import { from } from 'rxjs';

@Injectable()
export class CronServicePublisher {
  @Cron('0 0 0 * * *', {
    timeZone: process.env.CRON_TIMEZONE || 'UTC'
  })
  async handleCron() {
    const start = new Date();
    const startDate = start.toISOString();
    console.log('Running hourly job',startDate );
    try {
      await this.analyze();
    } catch (error) {
      // this.logger.error(`Failed to run analyze: ${error.message}`, error.stack);
    }
  }

  constructor(private appService: AppService) {}

  private async analyze(): Promise<void> {
    try{
      const result = await from(this.appService.notifySubscribers()).toPromise();
      console.log('Subscribers notified', result);
    }catch (error) {
      console.error(`Cron job error: ${error.message}`, error.stack);
      throw new HttpException(error.message, 500);
    }finally {
      console.log('Notification process completed.');
    }
    // this.appService.notifySubscribers().subscribe({
    //   next: (created) => {
    //     console.log('Subscribers notified', created);
    //   },
    //   error: (error) => {
    //     console.error(`Cron job error: ${error.message}`, error.stack);
    //     throw new HttpException(error.message, 500);
    //   },
    //   complete: () => {
    //     console.log('Notification process completed.');
    //   },
    // });
  }
}
